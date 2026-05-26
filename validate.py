"""
validate.py — convenience wrapper that reproduces the official local
synthetic-eval split (using `prepare_local_eval.py` from the dataset) and
scores a submission file against it.

Expected usage AFTER `train.py` + a predict run with the synth threshold:

    # produce the local ground truth (does not need GPU)
    AVITO_THRESHOLD_MS=1775606400000 python validate.py prepare

    # train + predict in local-validation mode (re-uses cache):
    AVITO_THRESHOLD_MS=1775606400000 python train.py
    AVITO_THRESHOLD_MS=1775606400000 python predict.py

    # score
    python validate.py score

`prepare` shells out to the dataset's own prepare_local_eval.py — it is
the canonical implementation of the synth split.
"""

import os
import subprocess
import sys

from src.calc_metric import calc_recall_at_160

DATA  = os.environ.get("AVITO_DATA",  "/data")
CACHE = os.environ.get("AVITO_CACHE", "/workspace/cache")
OUT   = os.environ.get("AVITO_OUT",   "/workspace/submission.csv")


def cmd_prepare():
    """Build local_eval.csv via the dataset's official script."""
    script = f"{DATA}/prepare_local_eval.py"
    if not os.path.exists(script):
        sys.exit(f"missing {script} — copy it from the competition dataset")
    subprocess.run(
        [
            "python", script,
            "--train", f"{DATA}/train_data/*.parquet",
            "--item-features", f"{DATA}/item_features.parquet",
            "--contact-eids", f"{DATA}/contact_eids.csv",
            "--out", f"{DATA}/local_eval.csv",
        ],
        check=True,
    )


def cmd_score():
    truth = f"{DATA}/local_eval.csv"
    if not os.path.exists(truth):
        sys.exit(f"missing {truth} — run `python validate.py prepare` first")
    calc_recall_at_160(OUT, truth)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "score"):
        print(__doc__)
        sys.exit(2)
    {"prepare": cmd_prepare, "score": cmd_score}[sys.argv[1]]()
