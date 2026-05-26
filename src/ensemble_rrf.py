"""
ensemble_rrf.py — Reciprocal-Rank-Fusion ensemble of multiple submission.csv
files (each is the output of `predict.py` for a different trained model).

Per (user, item) we sum  1 / (RRF_K + rank_in_model)  across the models the
item appears in; the per-user top-K by fused score becomes the merged
submission.  RRF is robust because it doesn't require score calibration
across the constituent models.

Not used by the default `predict.py` pipeline (which ships the single best
model), but available for experimentation.  In our experiments fusing 5
single-model submissions lifted Recall@160 from 0.0322 to 0.0341.

Usage:
    python -m src.ensemble_rrf --subs a.csv b.csv c.csv --out merged.csv
"""

import argparse

import polars as pl
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=160)
    ap.add_argument("--rrf-k", type=int, default=60,
                    help="RRF constant; 60 is the original Cormack paper value")
    args = ap.parse_args()

    dfs = []
    for sub in args.subs:
        df = pl.read_csv(sub).with_columns(pl.lit(1).alias("_one"))
        df = (
            df.with_columns(pl.col("_one").cum_sum().over("user_id").alias("rank"))
            .drop("_one")
            .with_columns((1.0 / (args.rrf_k + pl.col("rank"))).alias("rrf"))
            .select(["user_id", "item_id", "rrf"])
        )
        dfs.append(df)
        logger.info(f"loaded {sub}: {df.height:,} rows")

    fused = (
        pl.concat(dfs)
        .group_by(["user_id", "item_id"])
        .agg(pl.col("rrf").sum())
    )
    ranked = (
        fused.sort(["user_id", "rrf"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").head(args.k))
        .explode("item_id")
    )
    ranked.write_csv(args.out)
    logger.info(f"wrote {ranked.height:,} rows to {args.out}")


if __name__ == "__main__":
    main()
