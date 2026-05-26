"""
build_vocab.py — стадия 1 пайплайна: построение словаря объявлений.

Что делает:
  Сканирует все события (контакты пользователей) и составляет таблицу
  объявлений-кандидатов, на которых будет обучаться и предсказывать
  модель.

Почему нельзя взять `item_features.parquet` целиком?
  В нём 178 млн объявлений; больше половины — "мёртвые" (никто никогда
  с ними не взаимодействовал).  Хранить их в эмбеддинг-таблице — это
  десятки гигабайт зря; ранжировать тоже бессмысленно.

Фильтры словаря:
  1. vertical_id ∈ {0, 2, 3, 4, 5, 7} — это вертикали, по которым
     организаторы стратифицируют официальный eval (см. popular.py).
     Остальные 1, 6 — это "хвост" без покрытия в evalе.
  2. n_users ≥ 2 — у объявления должно быть минимум 2 уникальных
     юзера в pre-threshold событиях.  Условие совпадает с тем, что
     официальный prepare_local_eval требует от target-объявлений
     (минимум 2 уникальных юзера в synthetic train).

ВЫХОДЫ:
  $AVITO_CACHE/item_vocab.parquet
      Колонки: item_id, item_idx, vertical_id, category_ext_y,
               region_id_y, loc_id_y, sid_0_y..sid_3_y, n_users, log1p_pop.
      `item_idx` — целочисленный индекс [0, n_items), под который
      выделено место в эмбеддинг-таблицах модели.
  $AVITO_CACHE/feature_dims.json
      Cardinality каждой категориальной колонки — нужна модели для
      создания `nn.Embedding(dims[col], d)` правильного размера.
"""

import json
import os
import time
from pathlib import Path

import polars as pl
from loguru import logger

# Стандартное соглашение: данные приезжают в /data (можно
# переопределить переменной окружения), результаты пайплайна — в
# /workspace/cache.  Это позволяет docker-у работать с любыми
# точками монтирования.
DATA  = os.environ.get("AVITO_DATA",  "/data")
CACHE = os.environ.get("AVITO_CACHE", "/workspace/cache")

# 6 из 8 вертикалей — длинный хвост 1 и 6 не покрывается evalом.
USED_VERTICAL_IDS = [0, 2, 3, 4, 5, 7]

# По умолчанию словарь строится по ВСЕМ событиям (для реального
# сабмита).  Для воспроизведения локальной валидации задайте
# AVITO_THRESHOLD_MS = 1775606400000  (2026-04-08 00:00 UTC) —
# это тот же synth threshold, что использует официальный
# prepare_local_eval.py.
SYNTH_THRESHOLD_MS = int(os.environ.get("AVITO_THRESHOLD_MS", "0"))

# Минимум 2 уникальных юзера на объявление — фильтр шумовых
# одноразовых объявлений (совпадает с MIN_USERS_PER_ITEM из
# prepare_local_eval.py).
MIN_USERS_PER_ITEM = 2


def main():
    os.makedirs(CACHE, exist_ok=True)

    train_glob          = f"{DATA}/train_data/*.parquet"
    item_features_path  = f"{DATA}/item_features.parquet"

    # Если включён threshold-фильтр, оставляем только события до этой
    # точки — иначе ВСЕ события идут в словарь (для реального сабмита).
    threshold_filter = (
        pl.col("timestamp") < SYNTH_THRESHOLD_MS
        if SYNTH_THRESHOLD_MS > 0 else pl.lit(True)
    )
    if SYNTH_THRESHOLD_MS > 0:
        logger.info(f"pre-threshold mode: ts < {SYNTH_THRESHOLD_MS}")

    # ── Шаг 1. Сколько уникальных юзеров было у каждого объявления.
    # Это самый тяжёлый scan: ~6×10⁹ строк.  Используем `streaming`
    # engine, чтобы Polars не пытался держать всё в RAM.
    t0 = time.time()
    logger.info(f"scan: items with >= {MIN_USERS_PER_ITEM} unique users")
    item_pop = (
        pl.scan_parquet(train_glob)
        .filter(threshold_filter)
        .group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n_users"))
        .filter(pl.col("n_users") >= MIN_USERS_PER_ITEM)
        .collect(engine="streaming")
    )
    logger.info(f"  items with >=2 users: {item_pop.height:,} "
                f"(t={time.time()-t0:.1f}s)")

    # ── Шаг 2. Берём метаданные объявлений (вертикали, категории,
    # регионы, SID-коды) и оставляем только нужные вертикали.
    items_meta = pl.read_parquet(item_features_path).filter(
        pl.col("vertical_id").is_in(USED_VERTICAL_IDS)
    )
    logger.info(f"items in used verticals: {items_meta.height:,}")

    # ── Шаг 3. Inner join: и в нужной вертикали, И с ≥2 юзерами.
    vocab = items_meta.join(item_pop, on="item_id", how="inner").sort("item_id")
    vocab = vocab.with_columns(
        # Целочисленный item_idx ∈ [0, n_items).  Модель индексирует
        # эмбеддинги по нему, а raw `item_id` мы храним лишь для
        # финального вывода в submission.csv.
        pl.int_range(0, vocab.height, dtype=pl.Int32).alias("item_idx"),
        # log1p(n_users) используется для квантизации в pop-buckets
        # в train_model.py.  Сохраняем здесь, чтобы не пересчитывать.
        (pl.col("n_users").cast(pl.Float32) + 1.0).log().alias("log1p_pop"),
    )
    vocab.write_parquet(f"{CACHE}/item_vocab.parquet")
    logger.info(f"vocab → {CACHE}/item_vocab.parquet  "
                f"({vocab.height:,} items)")

    # ── Шаг 4. Сохраняем cardinality каждой колонки (для `nn.Embedding`).
    dims = {}
    for c in ("vertical_id", "category_ext_y", "region_id_y", "loc_id_y",
              "sid_0_y", "sid_1_y", "sid_2_y", "sid_3_y"):
        # +1, потому что нужна 1 свободная строка под "padding" для
        # ItemTower — на всякий случай, если на инференсе встретится
        # значение чуть больше, чем dims[col].
        dims[c] = int(vocab[c].max()) + 1
    dims["n_items"] = int(vocab.height)

    with open(f"{CACHE}/feature_dims.json", "w") as f:
        json.dump(dims, f, indent=2)
    logger.info(f"feature dims: {dims}")


if __name__ == "__main__":
    main()
