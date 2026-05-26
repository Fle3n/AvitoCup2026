"""
train_model.py — обучение TwoTower нейронной сети.

ЛОСС
----
Sampled-softmax с двумя источниками негативов в каждом батче:
  * In-batch negatives: для каждого пользователя из батча его positive
    item — это столбец `i`; все остальные `B-1` positive-item'ов в
    батче автоматически становятся "free" negatives.  С B=8192 это
    даёт 8191 негатив на каждый пример БЕСПЛАТНО (всего один matmul).
  * Random negatives: дополнительно сэмплируем `num_negatives` объявлений
    равномерно из каталога.  Они нужны, чтобы покрыть "длинный хвост",
    которого в батче может не оказаться (in-batch negatives слегка
    смещены к популярным).

ОПТИМИЗАТОР
-----------
fused AdamW + linear warmup + cosine decay.  Базовая комбинация для
стабильного обучения трансформера.  Wamup делает 500 шагов.

ПОЧЕМУ НЕТ `torch.amp.GradScaler`?
----------------------------------
GradScaler нужен ИСКЛЮЧИТЕЛЬНО при fp16 autocast, потому что у fp16
маленький экспоненциальный диапазон и градиенты часто underflow'ят.
Мы используем bf16 autocast — у него тот же экспоненциальный диапазон,
что и у fp32, поэтому GradScaler не нужен.  Хуже того, если использовать
GradScaler с bf16, scaler.scale накапливает значения, которые периодически
вызывают катастрофические расхождения лосса на границах эпох (мы это
прошли в v8 — loss взорвался от 0.3 до 130 000 в эпохе 4).

УСТОЙЧИВОСТЬ
------------
- `torch.nn.utils.clip_grad_norm_(..., 1.0)` после backward.  Срезает
  редкие выбросы градиентов без замедления нормального обучения.
- `pl.col(...).cum_sum().over(...)` для построения contact-position
  индекса делается ВЕКТОРИЗОВАННО (через numpy `repeat`/`diff`), а не
  python-циклом — первая версия с циклом тащила скорость до 1.5
  step/sec; вектор лифтнул до 25 step/sec.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .model import TwoTower, gather_feats


DATA  = os.environ.get("AVITO_DATA",  "/data")
CACHE = os.environ.get("AVITO_CACHE", "/workspace/cache")


def parse_args():
    """
    Гиперпараметры по умолчанию — те, на которых наш лучший single-
    model результат (Recall@160 = 0.0322 на local-eval split):
        d=128, batch=8192, 3 эпохи, lr=2e-3, num_negatives=8192.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=128,
                   help="Размерность item/user эмбеддингов")
    p.add_argument("--batch", type=int, default=8192,
                   help="Размер батча; служит ещё и числом in-batch негативов")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--num-negatives", type=int, default=8192,
                   help="Доп. равномерно-случайных негативов на шаг (поверх in-batch)")
    p.add_argument("--max-hist", type=int, default=48,
                   help="Длина последовательности для UserTower")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-user", type=int, default=4,
                   help="Сколько (history, target) пар сэмплируется из юзера за эпоху")
    p.add_argument("--out", type=str, default=f"{CACHE}/model.pt")
    return p.parse_args()


class TrainingData:
    """
    Хранит все CPU-тензоры, которые нужны на каждой итерации обучения.

    Главная идея: вместо того чтобы держать список python-списков
    (`List[List[int]]` для каждого юзера), мы используем РАГГЕД-ТЕНЗОР:
       items_flat  — 1D np.int32 со всеми item_idx'ами всех юзеров подряд.
       offsets     — 1D np.int64, длина n_users+1: события юзера `u`
                     лежат в `items_flat[offsets[u] : offsets[u+1]]`.
    Это даёт нам O(1)-доступ к истории любого юзера через два slice'а
    и поддерживает векторизованный sampler.
    """

    def __init__(self, max_hist: int):
        contact_eids = (
            pl.read_csv(f"{DATA}/contact_eids.csv")["mapped_eid"].to_list()
        )

        logger.info("loading vocab + feature dims...")
        vocab = pl.read_parquet(f"{CACHE}/item_vocab.parquet").sort("item_idx")
        with open(f"{CACHE}/feature_dims.json") as f:
            self.dims = json.load(f)
        self.n_items = int(self.dims["n_items"])

        # Per-item фичи (1D-тензоры длины n_items) — модель индексирует
        # их по item_idx за O(1).  Сохраняем как int64, потому что
        # `nn.Embedding` требует LongTensor.
        as_l = lambda c: torch.from_numpy(vocab[c].to_numpy().astype(np.int64))
        self.feat_vert = as_l("vertical_id")
        self.feat_cat  = as_l("category_ext_y")
        self.feat_reg  = as_l("region_id_y")
        self.feat_loc  = as_l("loc_id_y")
        self.feat_s0   = as_l("sid_0_y")
        self.feat_s1   = as_l("sid_1_y")
        self.feat_s2   = as_l("sid_2_y")
        self.feat_s3   = as_l("sid_3_y")

        # Бакетизация log1p(n_users) в 128 квантильных бинов.
        # Это даёт модели низкоразмерный сигнал популярности, который
        # сложно потерять при усреднении content-фич.
        log1p_pop = vocab["log1p_pop"].to_numpy().astype(np.float64)
        N_POP_BUCKETS = 128
        edges = np.sort(log1p_pop)[
            np.linspace(0, len(log1p_pop) - 1, N_POP_BUCKETS + 1, dtype=int)
        ]
        pop_bucket = np.clip(
            np.searchsorted(edges[1:-1], log1p_pop), 0, N_POP_BUCKETS - 1
        )
        self.feat_pop = torch.from_numpy(pop_bucket.astype(np.int64))
        logger.info(f"items: {self.n_items:,}, dims: {self.dims}")

        # ── Загружаем user-последовательности из build_user_seq.parquet.
        logger.info("loading user sequences...")
        seq = pl.read_parquet(f"{CACHE}/user_seq.parquet")
        logger.info(f"users with sequences: {seq.height:,}")

        # Для каждого юзера маркируем, на каких позициях у него был контакт.
        # Полученная булева маска позволит sampler'у выбирать ТОЛЬКО
        # contact-позиции как target'ы.
        contact_mask_per_user = seq.with_columns(
            pl.col("hist_eids")
            .list.eval(pl.element().is_in(contact_eids).cast(pl.Int8))
            .alias("contact_mask")
        )
        items_l = contact_mask_per_user["hist_items"].to_list()
        eids_l  = contact_mask_per_user["hist_eids"].to_list()
        cmask_l = contact_mask_per_user["contact_mask"].to_list()

        lens = np.fromiter((len(x) for x in items_l), dtype=np.int64,
                           count=len(items_l))
        total = int(lens.sum())
        items_flat = np.empty(total, dtype=np.int32)
        eids_flat  = np.empty(total, dtype=np.int8)
        cmask_flat = np.empty(total, dtype=np.int8)
        offsets    = np.empty(len(items_l) + 1, dtype=np.int64)
        offsets[0] = 0
        idx = 0
        for i, (a, b, c) in enumerate(zip(items_l, eids_l, cmask_l)):
            n = len(a)
            items_flat[idx:idx+n] = a
            eids_flat [idx:idx+n] = b
            cmask_flat[idx:idx+n] = c
            idx += n
            offsets[i+1] = idx
        self.items   = items_flat
        self.eids    = eids_flat
        self.cmask   = cmask_flat
        self.offsets = offsets
        self.n_users = len(items_l)
        self.max_hist = max_hist

        # ── Векторизованное построение contact-position индекса.
        # Это критично по скорости — оригинал с python-циклом тащил
        # 8 минут, векторизация — 200 мс.
        lengths = np.diff(offsets).astype(np.int64)
        # global_position → (user_id, local_position_within_user)
        local_pos = (
            np.arange(total, dtype=np.int64)
            - np.repeat(offsets[:-1], lengths)
        )
        user_ids_flat = np.repeat(
            np.arange(self.n_users, dtype=np.int64), lengths
        )
        # Оставляем только позиции, где было контактное событие, И
        # позиция >= 1 (нужна хотя бы 1 история перед target'ом).
        keep = (cmask_flat == 1) & (local_pos >= 1)
        self.contact_pos = np.stack(
            [user_ids_flat[keep], local_pos[keep]], axis=1
        )
        logger.info(
            f"contact-position training pairs: {len(self.contact_pos):,}"
        )

    def sample_batch(self, rng: np.random.Generator, B: int):
        """
        Векторизованный батч-сэмплер.  Возвращает четыре torch-тензора:
          hist:   (B, H) int64   — последние H item_idx'ов перед target'ом
          eids:   (B, H) int64   — eid'ы на тех же позициях
          mask:   (B, H) float32 — 1 на валидных позициях, 0 на padding
          target: (B,)   int64   — item_idx, который надо предсказать

        Историю переворачиваем (oldest=0, newest=H-1) — Transformer'у
        удобнее последовательность хронологически "слева направо".

        ─── Пример ───────────────────────────────────────────────────────
        У юзера всего 5 событий с eid'ами:  [view, view, CONTACT, view, CONTACT]
        и item_idx'ами:                      [  a ,   b ,    c   ,  d ,    e   ]
        (CONTACT — это то, что считается positive для обучения.)

        В contact_pos для этого юзера будут две позиции: index=2 (c) и
        index=4 (e), потому что на них был контакт И index >= 1
        (есть как минимум 1 событие до).

        Если sampler выбрал позицию index=4 (target=e) при H=3:
            target_global    = offsets[u] + 4
            k_back           = [1, 2, 3]
            positions        = target_global - [1, 2, 3]  → d, c, b
            hist_rev         = [d, c, b]   (от свежего к старому)
            hist (после ::-1)= [b, c, d]   (от старого к свежему)
            eids             = [view, CONTACT, view]
            mask             = [1, 1, 1]   (все позиции валидны)
            target           = e

        Если sampler выбрал index=2 (target=c) при H=3, то столбцов
        истории всего 2 (b, a), и в первой позиции (oldest) будет
        padding:
            positions        = [1, 0, -1]   → b, a, <invalid>
            valid            = [1, 1, 0]    (поэлементно position >= 0)
            hist (clipped)   = [<pad>, a, b]
            mask             = [0, 1, 1]

        Маска нужна attention'у в UserTower, чтобы он игнорировал pad.
        ────────────────────────────────────────────────────────────────
        """
        H = self.max_hist
        idx = rng.integers(0, len(self.contact_pos), size=B)
        chosen = self.contact_pos[idx]
        users   = chosen[:, 0]
        locals_ = chosen[:, 1]
        target_global = self.offsets[users] + locals_
        target = self.items[target_global].astype(np.int64)

        # Для каждой позиции k = 1..H берём позицию (target - k).
        # Получаем матрицу (B, H), где столбец 0 = newest_past, столбец H-1 = oldest.
        k_back     = np.arange(1, H + 1, dtype=np.int64)[None, :]
        positions  = target_global[:, None] - k_back
        user_start = self.offsets[users][:, None]
        # Маркер валидности: позиция должна лежать внутри истории юзера.
        valid      = positions >= user_start
        positions_clipped = np.maximum(positions, 0)

        hist_rev = self.items[positions_clipped]
        eids_rev = self.eids [positions_clipped].astype(np.int64)
        # Переворачиваем: oldest=column 0, newest=column H-1.
        hist = np.ascontiguousarray(hist_rev[:, ::-1]).astype(np.int64)
        eids = np.ascontiguousarray(eids_rev[:, ::-1])
        mask = np.ascontiguousarray(valid[:, ::-1]).astype(np.float32)
        return (
            torch.from_numpy(hist),
            torch.from_numpy(eids),
            torch.from_numpy(mask),
            torch.from_numpy(target),
        )

    def sample_negatives(self, n: int, rng: np.random.Generator):
        """
        Равномерно-случайные негативные item_idx'ы.

        Почему не popularity-weighted (стандартная Word2Vec техника)?
        - `torch.multinomial` физически не умеет работать с категориями
          > 2^24 = 16.7M, а у нас 1.2×10⁸ объявлений.
        - In-batch negatives УЖЕ дают сильный popularity bias (target'ы
          в батче скоррелированы с реальной популярностью), так что
          random pool можно оставить uniform для покрытия хвоста.
        """
        neg = rng.integers(0, self.n_items, size=n).astype(np.int64)
        return torch.from_numpy(neg)


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    td = TrainingData(args.max_hist)
    device = torch.device("cuda")
    contact_eids = (
        pl.read_csv(f"{DATA}/contact_eids.csv")["mapped_eid"].to_list()
    )
    # +16 запас — eid'ы в данных — UInt32, но реально все маленькие
    # (≤ 16 у нас, плюс зазор на padding-row "0" в эмбеддинге).
    n_eids = max(contact_eids) + 16

    model = TwoTower(
        dims=td.dims, d=args.d, max_hist=args.max_hist,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, n_eids=n_eids,
    ).to(device)
    logger.info(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Переносим feature lookup-таблицы на GPU один раз — потом в каждом
    # шаге индексирование делается ZERO COPY.
    for name in ("feat_vert", "feat_cat", "feat_reg", "feat_loc",
                 "feat_s0", "feat_s1", "feat_s2", "feat_s3", "feat_pop"):
        setattr(td, name, getattr(td, name).to(device))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,  # fused-AdamW значительно быстрее на современных GPU
    )

    n_pairs_per_epoch = len(td.contact_pos) * args.samples_per_user
    steps_per_epoch = max(1, n_pairs_per_epoch // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, total_steps // 20)
    logger.info(
        f"steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}"
    )

    def lr_at(step: int) -> float:
        """Linear warmup, затем cosine decay до lr/50."""
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        floor = args.lr / 50.0
        return floor + (args.lr - floor) * cosine

    # Lookup-словарь для `gather_feats` — один источник истины для формата
    # фич, шарится с predict_model через `src.model.gather_feats`.
    feat_lut = {
        "vert": td.feat_vert, "cat": td.feat_cat, "reg": td.feat_reg,
        "loc":  td.feat_loc,  "s0":  td.feat_s0,  "s1":  td.feat_s1,
        "s2":   td.feat_s2,   "s3":  td.feat_s3,  "pop": td.feat_pop,
    }

    step = 0
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        running_loss = 0.0
        for it in range(steps_per_epoch):
            hist, hist_eid, hist_mask, target = td.sample_batch(rng, args.batch)
            negs = td.sample_negatives(args.num_negatives, rng)

            hist      = hist.to(device, non_blocking=True)
            hist_eid  = hist_eid.to(device, non_blocking=True)
            hist_mask = hist_mask.to(device, non_blocking=True)
            target    = target.to(device, non_blocking=True)
            negs      = negs.to(device, non_blocking=True)

            hist_feats   = gather_feats(feat_lut, hist)
            target_feats = gather_feats(feat_lut, target)
            negs_feats   = gather_feats(feat_lut, negs)

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                u   = model.user_embed(hist_feats, hist_eid, hist_mask)
                pos = model.item(target_feats)
                neg = model.item(negs_feats)
                # logits[i, j]:
                #   j ∈ [0, B)     — in-batch positives (j-й positive item)
                #   j ∈ [B, B+N)   — N "free" random negatives
                # Корректный ответ для строки i — это column i.
                logits = torch.cat([u @ pos.t(), u @ neg.t()], dim=1)
                labels = torch.arange(args.batch, device=device)
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += float(loss.detach())
            step += 1
            if (step + 1) % 100 == 0:
                logger.info(
                    f"epoch {epoch+1}/{args.epochs} step {step+1} "
                    f"loss={running_loss/100:.4f}  lr={lr_at(step):.2e}  "
                    f"t={time.time()-ep_t0:.1f}s"
                )
                running_loss = 0.0

        logger.info(f"epoch {epoch+1} done in {time.time()-ep_t0:.1f}s")
        # Чекпоинт после каждой эпохи — даже если следующая упадёт,
        # у нас останется работоспособная модель.
        torch.save({
            "model":  model.state_dict(),
            "args":   vars(args),
            "dims":   td.dims,
            "n_eids": n_eids,
        }, args.out)
        logger.info(f"saved {args.out}")

    logger.info("train_model.py DONE.")


if __name__ == "__main__":
    train(parse_args())
