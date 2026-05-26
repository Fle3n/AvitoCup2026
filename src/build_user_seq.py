"""
build_user_seq.py — стадия 2 пайплайна: history-последовательности
для обучения.

ЗАЧЕМ ЭТО НУЖНО
---------------
Для обучения SASRec-подобной модели нам нужно для каждого пользователя
хронологически отсортированный список последних K событий (item_idx,
event_type, timestamp).  Из них в train_model.py мы будем сэмплировать
позиции с контактом как target'ы, а всё, что было ДО них — как
"history-входы" для UserTower.

ПОЧЕМУ ПОПАРТИЦИОННО
--------------------
Полный union train_data + eval_user_events — это ~6 млрд событий.
Глобальный sort + group_by на этом объёме падает по OOM даже на
500 ГБ RAM (проверено: polars 1.41 panics: "maximum length reached").

train_data удобно партиционирована по `user_id % 100`, поэтому ВСЕ
события одного пользователя лежат в одной партиции — мы можем обработать
каждую из 100 партиций независимо, аггрегируя их per-user без
глобального sort'а.  Peak RSS под 20 ГБ, время ~3-5 минут.

ВАЖНЫЙ ФИЛЬТР
-------------
Оставляем только пользователей, у которых ХОТЯ БЫ ОДНО событие — контакт
(eid из contact_eids.csv).  Если у пользователя только просмотры — у
него нет target'ов для обучения, нет смысла занимать им место.

ВЫХОД
-----
  $AVITO_CACHE/user_seq.parquet
      Колонки: user_id (UInt32), hist_items (List[Int32]),
               hist_eids (List[UInt32]), hist_ts (List[Int64]).
      Каждая запись — один пользователь и его последние MAX_HISTORY
      событий (item_idx уже отображены через item_vocab).
"""

import gc
import os
import time
from pathlib import Path

import polars as pl
from loguru import logger

DATA  = os.environ.get("AVITO_DATA",  "/data")
CACHE = os.environ.get("AVITO_CACHE", "/workspace/cache")

# 0 = брать все события (по умолчанию для реального сабмита).
# AVITO_THRESHOLD_MS=1775606400000 — для локальной валидации.
SYNTH_THRESHOLD_MS = int(os.environ.get("AVITO_THRESHOLD_MS", "0"))

MAX_HISTORY = 64
# Кэп на число пользователей в обучении — чтобы word_seq.parquet
# не разрастался без надобности.  У нас типично ~8 млн пользователей с
# контактами; модель сходится и за выборку из 5 млн.
TRAIN_USER_CAP = 5_000_000


def main():
    contact_eids = (
        pl.read_csv(f"{DATA}/contact_eids.csv")["mapped_eid"].to_list()
    )
    logger.info(f"contact eids: {contact_eids}")

    # Маппинг item_id → item_idx, чтобы хранить ОДНОБАЙТНЫЕ индексы
    # в hist_items вместо громоздких UInt32-ID'ов.
    id_map = pl.read_parquet(
        f"{CACHE}/item_vocab.parquet", columns=["item_id", "item_idx"]
    )
    logger.info(f"vocab id_map: {id_map.height:,}")

    threshold_filter = (
        pl.col("timestamp") < SYNTH_THRESHOLD_MS
        if SYNTH_THRESHOLD_MS > 0 else pl.lit(True)
    )

    parts: list[pl.DataFrame] = []
    # `part_[0-9]*` — намеренно исключает `part_eval.parquet`-симлинк,
    # если кто-то его создавал в /workspace/data/train_data во время
    # экспериментов.  Eval-юзеров мы обрабатываем отдельно в predict.
    files = sorted(Path(f"{DATA}/train_data").glob("part_[0-9]*.parquet"))
    logger.info(f"train partitions: {len(files)}")

    for i, p in enumerate(files):
        t0 = time.time()
        df = (
            pl.scan_parquet(str(p))
            .filter(threshold_filter)
            # Inner join с vocab → автоматически выкидываем item'ы вне
            # USED_VERTICAL_IDS и < MIN_USERS_PER_ITEM.
            .join(id_map.lazy(), on="item_id", how="inner")
            .sort(["user_id", "timestamp"])
            .group_by("user_id", maintain_order=True)
            .agg(
                pl.col("item_idx").tail(MAX_HISTORY).alias("hist_items"),
                pl.col("eid").tail(MAX_HISTORY).alias("hist_eids"),
                pl.col("timestamp").tail(MAX_HISTORY).alias("hist_ts"),
                pl.col("eid").is_in(contact_eids).any().alias("has_contact"),
            )
            # Фильтр: оставляем только пользователей с хотя бы 1 контактом
            # в pre-threshold окне — у остальных нет обучающего сигнала.
            .filter(pl.col("has_contact"))
            .drop("has_contact")
            .collect(engine="streaming")
        )
        parts.append(df)
        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(files)} ({time.time()-t0:.1f}s/part)")
        gc.collect()

    all_users = pl.concat(parts)
    del parts; gc.collect()
    logger.info(f"raw train users with contacts: {all_users.height:,}")

    # Сэмплируем до TRAIN_USER_CAP — это даёт нам управляемый размер
    # in-memory тензора в train_model.py.  Случайный seed=42 для
    # воспроизводимости.
    if all_users.height > TRAIN_USER_CAP:
        all_users = all_users.sample(n=TRAIN_USER_CAP, seed=42)
        logger.info(f"sampled to {all_users.height:,} train users")

    out = f"{CACHE}/user_seq.parquet"
    all_users.write_parquet(out)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
