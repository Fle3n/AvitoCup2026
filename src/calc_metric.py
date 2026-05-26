"""
calc_metric.py — Recall@160 в точности так, как определила метрика
организаторами.

Для каждого пользователя с >=1 целевым объявлением:
    recall_u = |predicted_u ∩ targets_u| / |targets_u|
Итоговая метрика — среднее по пользователям с targets.

Особенности:
  * Пользователи, у которых есть targets, но НЕТ предсказаний, дают 0
    (это соответствует "missing users получают score 0" из спеки).
  * Пользователи, попавшие в predictions, но не имеющие targets,
    игнорируются (на них не считается recall).
  * Дубликаты в predictions удаляются — реально работает уникальная
    `(user_id, item_id)` пара.

Использование:
    python -m src.calc_metric --pred submission.csv --truth local_eval.csv
"""

import argparse

import polars as pl
from loguru import logger


def calc_recall_at_160(pred_path: str, truth_path: str) -> float:
    pred  = pl.read_csv(pred_path ).select(["user_id", "item_id"]).unique()
    truth = pl.read_csv(truth_path).select(["user_id", "item_id"]).unique()

    # Сколько targets'ов у каждого пользователя.
    n_targets = (
        truth.group_by("user_id").len().rename({"len": "n_targets"})
    )
    # Сколько pred попадает в targets для каждого пользователя.
    inter = (
        pred.join(truth, on=["user_id", "item_id"], how="inner")
        .group_by("user_id").len().rename({"len": "n_hits"})
    )
    per_user = (
        n_targets.join(inter, on="user_id", how="left")
        # left join даёт null для юзеров без хитов — заменяем на 0.
        .with_columns(pl.col("n_hits").fill_null(0))
        .with_columns(
            (pl.col("n_hits").cast(pl.Float64)
             / pl.col("n_targets").cast(pl.Float64)).alias("recall")
        )
    )
    score = float(per_user["recall"].mean())
    logger.info(
        f"users with targets: {per_user.height:,}  |  "
        f"users in pred: {pred['user_id'].n_unique():,}  |  "
        f"Recall@160 = {score:.6f}"
    )
    return score


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred",  required=True, help="path to submission.csv")
    p.add_argument("--truth", required=True, help="path to local_eval.csv")
    a = p.parse_args()
    calc_recall_at_160(a.pred, a.truth)
