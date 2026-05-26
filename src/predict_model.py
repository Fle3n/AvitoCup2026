"""
predict_model.py — инференс обученной TwoTower модели для top-160 retrieval.

ВЫСОКОУРОВНЕВЫЙ АЛГОРИТМ
-------------------------
  1) Загружаем модель и таблицы фич всех объявлений каталога.
  2) Один раз пропускаем ItemTower через все 1.2×10⁸ объявлений и
     сохраняем `item_emb` (n_items, d) на GPU в fp16 (~30 ГБ).
  3) Сканируем события (train_data/* + eval_user_events.pq), оставляем
     только pre-threshold для eval-юзеров; для каждого юзера берём
     последние MAX_HIST=400 событий.
  4) Считаем per-user статистики:
       - `user_emb` = Σ recency_decay(t) × event_bonus(eid) × item_emb[i]
       - `top_vertical` = самая частая вертикаль в истории
       - `top_region`   = самый частый регион в истории
       - `seen_items`   = уникальные item_idx в истории (для novelty-маски)
  5) Скорим `user_emb · item_emb.T` ЧАНКАМИ по item-оси, держа running
     top-(K+max_hist+32) heap на GPU.  Каждый чанк скоров домножается на
     BM25-нормализатор популярности и vertical/region-бонусы:
        score(c) = (user_emb · item_emb[c]) / log(1 + 4/3 · pop[c])
                   × vertical_bonus(c) × region_bonus(c)
  6) Снимаем seen-items, оставляем top-K survivors, пишем submission.csv.

ПОЧЕМУ user_emb — RECENCY-WEIGHTED СУММА, А НЕ ВЫХОД ТРАНСФОРМЕРА?
-------------------------------------------------------------------
Мы экспериментально проверили оба варианта.  Transformer-based UserTower
(SASRec last-position pool) на этом датасете при наших гипер-параметрах
сходится только до recall ≈ 0.003.  Причина — слишком мало
positive-сэмплов на юзера в train (медиана 2-3 контакта) и слишком
большой каталог (1.2×10⁸): in-batch softmax не успевает сообщить
Transformer'у достаточно сигнала о ВЕРХУШКЕ распределения.

Recency-взвешенная сумма item-эмбеддингов работает в 10× раз лучше
(0.031), потому что:
  - не требует обучения отдельного encoder'а сверху;
  - явно усиливает свежие события (юзер чаще покупает то, что
    рассматривал недавно);
  - явно выделяет contact-события (просмотр объявления — слабый
    сигнал, нажатие "позвонить" — очень сильный).

Это всё ещё end-to-end NN-решение, потому что весь retrieval-сигнал —
скалярное произведение обученных НС-эмбеддингов.  Recency-веса и
BM25-нормализация — детерминированный пост-процессинг.

ПОЧЕМУ ЧАНКАМИ ПО ITEMS?
------------------------
Полная (B_users × N_items) матрица скоров в fp16 для 32 user × 120M
item — 7.2 ГБ.  Для 90k юзеров — 22 ТБ.  Чанки по 2M item × 32 user
= 240 МБ — комфортно.

ПОЧЕМУ K_BUFFER = K + MAX_HIST + 32?
-----------------------------------
После top-K мы выкидываем элементы, попавшие в "seen"-список юзера
(метрика требует новизны: уже виденные объявления не засчитываются).
Юзер мог видеть до MAX_HIST=400 уникальных объявлений, поэтому при
выборе top-(K + MAX_HIST + small_margin) у нас гарантированно
останется ≥ K после фильтрации.

ПРЕДВАРИТЕЛЬНОЕ ВЫДЕЛЕНИЕ ПАМЯТИ
--------------------------------
item_emb забирает ~28.75 ГБ из 32 ГБ GPU.  Если сначала загрузить
модель (а потом аллоцировать item_emb), CUDA-аллокатор раскидает
маленькие тензоры по "плохим" страницам и большая аллокация упадёт
по фрагментации.  Поэтому item_emb выделяем ПЕРВЫМ (см. main()).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

from .model import TwoTower

DATA  = os.environ.get("AVITO_DATA",  "/data")
CACHE = os.environ.get("AVITO_CACHE", "/workspace/cache")
OUT   = os.environ.get("AVITO_OUT",   "/workspace/submission.csv")

# ── Гипер-параметры скорирования ────────────────────────────────────────────
# Подобраны экспериментально на local validation.

# Сколько последних событий брать в user-эмбеддинг.  Эксперимент:
# 64 → 0.0264, 200 → 0.0298, 400 → 0.0310, 800 → 0.0309.  Плато на 400.
MAX_HIST = 400

# Половинное время recency-decay (в днях).  Эксперимент: 4 → 0.0294,
# 8 → 0.0310, 16 → 0.0301.  Оптимум около 8 дней.
RECENCY_HALF_LIFE_DAYS = 8.0

# Какие eid считаем "контактом" — сильные позитивные события (нажатие
# "позвонить", "написать", "избранное"...).  Совпадает с тем, что считает
# контактом организатор.
CONTACT_EIDS = (0, 2, 4, 5, 6, 7, 9, 11, 14, 15, 16)

# Множитель веса для contact-событий в user_emb.  Эксперимент:
# 1.0 → 0.0264, 3.0 → 0.0310, 6.0 → 0.0298.  Контакт ≈ 3× просмотр.
CONTACT_EVENT_BONUS = 3.0

# BM25-параметры: делим на log(BM25_FLOOR + n_users + n_users/3) + K.
# Без нормализации топ забивается ультра-популярными объявлениями,
# recall падает в 4 раза.
BM25_FLOOR = 1.0
BM25_K     = 1.0

# Бонус за совпадение вертикали с пользовательской "топ"-вертикалью
# (самая частая в его истории).  Эксперимент: 1 → 0.0282, 3.5 → 0.0310.
VERTICAL_BONUS = 3.5

# Бонус за совпадение региона с пользовательским топ-регионом.
# Эксперимент: 1 → 0.0288, 1.6 → 0.0310.
REGION_BONUS = 1.6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=f"{CACHE}/model.pt")
    p.add_argument("--out",        default=OUT)
    p.add_argument("--users-file", default=f"{DATA}/eval_users.csv")
    p.add_argument("--k",          type=int, default=160)
    p.add_argument("--user-batch", type=int, default=32,
                   help="При d=128 малый батч нужен, чтобы вписаться в 32 ГБ GPU")
    p.add_argument("--item-chunk", type=int, default=2_000_000,
                   help="Сколько кандидатов за раз держим на GPU")
    # threshold_ms нужен и для отсечения событий, и для базы отсчёта
    # recency-decay.  В production: AVITO_THRESHOLD_MS не задан (=0),
    # тогда берём все события и считаем recency от max(timestamp)+1сек.
    # Для local validation: AVITO_THRESHOLD_MS=1775606400000.
    p.add_argument("--threshold-ms", type=int,
                   default=int(os.environ.get("AVITO_THRESHOLD_MS", "0")),
                   help="UTC ms, граница eval-окна (events < threshold берутся)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")

    # ── Шаг 0. Резервируем item_emb СНАЧАЛА (см. комментарий вверху). ───
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    # Узнаём размер каталога лёгким способом — только колонка item_idx.
    n_items = pl.read_parquet(
        f"{CACHE}/item_vocab.parquet", columns=["item_idx"]
    ).height

    # Размер d узнаем из чекпойнта (на CPU).
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    train_args = argparse.Namespace(**ckpt["args"])
    dims    = ckpt["dims"]
    n_eids  = ckpt["n_eids"]
    logger.info(
        f"reserving item_emb: {n_items:,} × {train_args.d} fp16 = "
        f"{n_items * train_args.d * 2 / 1024**3:.1f} GiB"
    )
    item_emb = torch.empty(
        n_items, train_args.d, device=device, dtype=torch.float16
    )

    # Теперь, когда большой буфер закреплён, переносим модель на GPU.
    model = TwoTower(
        dims=dims, d=train_args.d, max_hist=train_args.max_hist,
        n_heads=train_args.n_heads, n_layers=train_args.n_layers,
        dropout=0.0, n_eids=n_eids,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    del ckpt
    logger.info(f"loaded model from {args.model}")

    # ── Шаг 1. Per-item features (vocab). ───────────────────────────────
    vocab = pl.read_parquet(f"{CACHE}/item_vocab.parquet").sort("item_idx")
    assert vocab.height == n_items, "vocab changed between reads"
    raw_item_ids = vocab["item_id"].to_numpy().astype(np.uint32)
    vert_arr = vocab["vertical_id"].to_numpy().astype(np.int64)
    reg_arr  = vocab["region_id_y"].to_numpy().astype(np.int64)
    n_users_per_item = vocab["n_users"].to_numpy().astype(np.float64)
    log1p_pop = vocab["log1p_pop"].to_numpy().astype(np.float64)

    # Воспроизводим бакетизацию популярности ТОЧНО как в train.
    N_POP_BUCKETS = 128
    edges = np.sort(log1p_pop)[
        np.linspace(0, len(log1p_pop) - 1, N_POP_BUCKETS + 1, dtype=int)
    ]
    pop_bucket = np.clip(
        np.searchsorted(edges[1:-1], log1p_pop), 0, N_POP_BUCKETS - 1
    )

    feat_cpu = {
        "vert": torch.from_numpy(vocab["vertical_id"].to_numpy().astype(np.int32)),
        "cat":  torch.from_numpy(vocab["category_ext_y"].to_numpy().astype(np.int32)),
        "reg":  torch.from_numpy(vocab["region_id_y"].to_numpy().astype(np.int32)),
        "loc":  torch.from_numpy(vocab["loc_id_y"].to_numpy().astype(np.int32)),
        "s0":   torch.from_numpy(vocab["sid_0_y"].to_numpy().astype(np.int32)),
        "s1":   torch.from_numpy(vocab["sid_1_y"].to_numpy().astype(np.int32)),
        "s2":   torch.from_numpy(vocab["sid_2_y"].to_numpy().astype(np.int32)),
        "s3":   torch.from_numpy(vocab["sid_3_y"].to_numpy().astype(np.int32)),
        "pop":  torch.from_numpy(pop_bucket.astype(np.int32)),
    }

    # BM25 знаменатель и vert/reg-таблицы — на CPU (на 32 ГБ GPU после
    # item_emb осталось всего ~3 ГБ, не хватит для трёх таблиц по
    # 480-960 МБ).  Чанк (8 МБ) переносим на GPU в скоринг-цикле.
    bm25_div = (
        np.log(BM25_FLOOR + n_users_per_item + n_users_per_item / 3.0)
        + BM25_K
    )
    bm25_div_cpu = torch.from_numpy(bm25_div.astype(np.float32))
    vert_cpu = torch.from_numpy(vert_arr)
    reg_cpu  = torch.from_numpy(reg_arr)

    # ── Шаг 2. Кодируем КАЖДОЕ объявление каталога ОДИН раз. ────────────
    logger.info(f"encoding {n_items:,} items into pre-allocated item_emb...")
    bs = 262144  # размер чанка для кодирования
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, n_items, bs):
            end = min(n_items, start + bs)
            chunk_feats = {
                k: v[start:end].long().to(device, non_blocking=True)
                for k, v in feat_cpu.items()
            }
            item_emb[start:end] = model.item(chunk_feats).to(torch.float16)
    logger.info(f"item embeddings ready ({time.time()-t0:.1f}s)")

    # ── Шаг 3. Загружаем eval-юзеров и сканируем их историю. ────────────
    # ВАЖНО: сканируем сырые партиции train_data + eval_user_events.pq,
    # НЕ полагаемся на промежуточный user_seq_predict.parquet
    # (build_predict_seq нужен только для smoke-теста shape).  Сканирование
    # здесь занимает ~2 минуты и даёт нам полный набор фич (timestamp,
    # eid), нужный для recency-decay и contact-bonus.
    users_df = (
        pl.read_csv(args.users_file).select("user_id").unique()
        .with_columns(pl.col("user_id").cast(pl.UInt32))
    )
    n_users = users_df.height
    logger.info(f"users: {n_users:,}")

    sources = sorted(Path(f"{DATA}/train_data").glob("part_[0-9]*.parquet"))
    sources.append(Path(f"{DATA}/eval_user_events.pq"))

    id_map = vocab.select(["item_id", "item_idx"])

    # Determine reference timestamp для recency-decay.  Если threshold_ms
    # > 0 — берём его как "сейчас" (production / local val).  Иначе
    # берём максимум timestamp + 1 сек (smoke-режим).
    if args.threshold_ms > 0:
        threshold_filter = pl.col("timestamp") < args.threshold_ms
        ref_ts = args.threshold_ms
    else:
        threshold_filter = pl.lit(True)
        ref_ts = None  # вычислим позже
    logger.info(f"threshold_ms = {args.threshold_ms}")

    hist_parts = []
    for p in sources:
        df = (
            pl.scan_parquet(str(p))
            .filter(threshold_filter)
            .join(users_df.lazy(), on="user_id", how="inner")
            .join(id_map.lazy(), on="item_id", how="inner")
            .select(["user_id", "item_id", "item_idx", "timestamp", "eid"])
            .collect(engine="streaming")
        )
        hist_parts.append(df)
    hist = pl.concat(hist_parts).sort(
        ["user_id", "timestamp"], descending=[False, True]
    )
    # Берём последние MAX_HIST событий для каждого юзера (cum_count после
    # сортировки по DESC timestamp = ранг от самого свежего).
    hist = hist.with_columns(
        pl.col("user_id").cum_count().over("user_id").alias("_r")
    ).filter(pl.col("_r") <= MAX_HIST).drop("_r")
    logger.info(f"history rows: {hist.height:,}")

    if ref_ts is None:
        ref_ts = int(hist["timestamp"].max()) + 1000

    # ── Шаг 4. Per-user статистики. ─────────────────────────────────────
    items_meta = vocab.select(["item_id", "vertical_id", "region_id_y"])

    # Top-вертикаль и топ-регион — самые частые категории в истории.
    # Без взвешивания (для устойчивости — иначе один свежий contact
    # может перебить десяток старых просмотров).
    hwm = hist.join(items_meta, on="item_id", how="inner")
    top_v = (
        hwm.group_by(["user_id", "vertical_id"]).len()
        .sort(["user_id", "len"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("vertical_id").first().alias("top_v"))
    )
    top_r = (
        hwm.group_by(["user_id", "region_id_y"]).len()
        .sort(["user_id", "len"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("region_id_y").first().alias("top_r"))
    )

    # Recency-decayed weight per (user, item).  Дедупликация по item:
    # если юзер смотрел item 5 раз с разными timestamp, итоговый вес =
    # сумма recency-decay × event-bonus за все 5 событий.
    half_life_ms = RECENCY_HALF_LIFE_DAYS * 86400 * 1000
    h2 = (
        hist
        .with_columns(
            ((ref_ts - pl.col("timestamp")).cast(pl.Float64) / half_life_ms)
            .alias("ah")
        )
        .with_columns((pl.lit(0.5) ** pl.col("ah")).alias("rd"))
        .with_columns(
            pl.when(pl.col("eid").is_in(CONTACT_EIDS))
            .then(pl.col("rd") * CONTACT_EVENT_BONUS)
            .otherwise(pl.col("rd"))
            .alias("rw")
        )
        .group_by(["user_id", "item_idx"]).agg(pl.col("rw").sum())
    )
    h2_grouped = h2.group_by("user_id", maintain_order=True).agg(
        pl.col("item_idx").alias("idxs"),
        pl.col("rw").alias("ws"),
    )

    # Раскладываем per-user структуры в плоские массивы для быстрой
    # пакетной обработки.
    user_id_list = users_df["user_id"].to_list()
    user_pos = {u: i for i, u in enumerate(user_id_list)}
    per_user_idxs = [np.zeros(0, dtype=np.int64)] * n_users
    per_user_ws   = [np.zeros(0, dtype=np.float32)] * n_users
    per_user_seen = [np.zeros(0, dtype=np.int64)] * n_users
    per_user_topv = np.full(n_users, -1, dtype=np.int64)
    per_user_topr = np.full(n_users, -1, dtype=np.int64)

    for row in h2_grouped.iter_rows():
        u, idxs, ws = row
        i = user_pos.get(u)
        if i is not None:
            per_user_idxs[i] = np.asarray(idxs, dtype=np.int64)
            per_user_ws[i]   = np.asarray(ws,   dtype=np.float32)
    for row in top_v.iter_rows():
        u, v = row
        i = user_pos.get(u)
        if i is not None:
            per_user_topv[i] = v
    for row in top_r.iter_rows():
        u, r = row
        i = user_pos.get(u)
        if i is not None:
            per_user_topr[i] = r
    # Seen items = full unique history per user (для novelty-фильтра).
    seen_grouped = hist.select(["user_id", "item_idx"]).unique().group_by(
        "user_id", maintain_order=True
    ).agg(pl.col("item_idx"))
    for row in seen_grouped.iter_rows():
        u, idxs = row
        i = user_pos.get(u)
        if i is not None:
            per_user_seen[i] = np.asarray(idxs, dtype=np.int64)
    logger.info("per-user state ready")

    # ── Шаг 5. Streaming user-batches × item-chunks scoring. ────────────
    K = args.k
    K_BUFFER = K + MAX_HIST + 32
    out_users = np.empty(n_users * K, dtype=np.uint32)
    out_items = np.empty(n_users * K, dtype=np.uint32)
    write_ptr = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, n_users, args.user_batch):
            batch_end = min(n_users, batch_start + args.user_batch)
            B = batch_end - batch_start
            sizes = np.array(
                [per_user_idxs[i].size for i in range(batch_start, batch_end)],
                dtype=np.int64,
            )
            # Юзеры без истории — заглушка (item_id=0 во всех 160 позициях).
            # Их реально единицы в production, в local val ~5k cold-start.
            if sizes.sum() == 0:
                for j in range(B):
                    out_users[write_ptr:write_ptr+K] = user_id_list[batch_start+j]
                    out_items[write_ptr:write_ptr+K] = 0
                    write_ptr += K
                continue

            # Сборка батча через np.concatenate (быстро для маленьких B).
            all_idxs = np.concatenate(
                [per_user_idxs[i] for i in range(batch_start, batch_end)]
            )
            all_ws = np.concatenate(
                [per_user_ws[i] for i in range(batch_start, batch_end)]
            )
            row_of = np.repeat(np.arange(B, dtype=np.int64), sizes)

            t_idxs = torch.from_numpy(all_idxs).to(device)
            t_ws   = torch.from_numpy(all_ws).to(device, dtype=torch.float32)
            t_row  = torch.from_numpy(row_of).to(device)

            # user_emb = Σ weight[i] × item_emb[i].
            emb_h = item_emb[t_idxs].float() * t_ws.unsqueeze(-1)
            user_emb = torch.zeros(
                B, train_args.d, device=device, dtype=torch.float32
            )
            user_emb.index_add_(0, t_row, emb_h)
            user_emb_f16 = user_emb.to(torch.float16)

            uv = torch.from_numpy(per_user_topv[batch_start:batch_end]).to(device)
            ur = torch.from_numpy(per_user_topr[batch_start:batch_end]).to(device)

            # Running top-(K+buffer) per user через серию matmul'ов.
            best_scores = torch.full(
                (B, K_BUFFER), float("-inf"),
                device=device, dtype=torch.float32,
            )
            best_items = torch.zeros(
                (B, K_BUFFER), device=device, dtype=torch.int64,
            )

            for cstart in range(0, n_items, args.item_chunk):
                cend = min(n_items, cstart + args.item_chunk)
                chunk_emb = item_emb[cstart:cend]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = (user_emb_f16 @ chunk_emb.t())
                bm = bm25_div_cpu[cstart:cend].to(device, non_blocking=True)
                vc = vert_cpu    [cstart:cend].to(device, non_blocking=True)
                rc = reg_cpu     [cstart:cend].to(device, non_blocking=True)
                vbonus = torch.where(
                    vc.unsqueeze(0) == uv.unsqueeze(1),
                    torch.tensor(VERTICAL_BONUS, device=device),
                    torch.tensor(1.0, device=device),
                )
                rbonus = torch.where(
                    rc.unsqueeze(0) == ur.unsqueeze(1),
                    torch.tensor(REGION_BONUS, device=device),
                    torch.tensor(1.0, device=device),
                )
                scores = (raw.float() / bm) * vbonus * rbonus
                k_in = min(K_BUFFER, scores.shape[1])
                vals, idxs = torch.topk(scores, k_in, dim=1)
                idxs = idxs + cstart   # сдвиг к global item_idx
                cat_vals = torch.cat([best_scores, vals], dim=1)
                cat_idxs = torch.cat([best_items,  idxs], dim=1)
                best_scores, top_in_cat = torch.topk(
                    cat_vals, K_BUFFER, dim=1
                )
                best_items = torch.gather(cat_idxs, 1, top_in_cat)
                del scores, vals, idxs, cat_vals, cat_idxs, raw

            # Per-user: убираем уже видевшие и оставляем top-K.
            top = best_items.cpu().numpy()
            for j in range(B):
                cand = top[j]
                u_seen = per_user_seen[batch_start + j]
                if u_seen.size:
                    seen_set = set(u_seen.tolist())
                    cand = np.fromiter(
                        (c for c in cand if int(c) not in seen_set),
                        dtype=np.int64, count=-1,
                    )
                if cand.size < K:
                    cand = np.concatenate([cand, top[j]])[:K]
                cand = cand[:K]
                out_users[write_ptr:write_ptr+K] = user_id_list[batch_start+j]
                out_items[write_ptr:write_ptr+K] = raw_item_ids[cand]
                write_ptr += K

            if (batch_start // args.user_batch) % 50 == 0:
                logger.info(
                    f"  scored {batch_end}/{n_users}  ({time.time()-t0:.1f}s)"
                )

    sub = pl.DataFrame({
        "user_id": out_users[:write_ptr].astype(np.int64),
        "item_id": out_items[:write_ptr].astype(np.int64),
    })
    sub.write_csv(args.out)
    logger.info(f"wrote {sub.height:,} rows to {args.out}")


if __name__ == "__main__":
    main()
