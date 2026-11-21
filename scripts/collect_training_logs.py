"""Copy the training records of our own systems from exp/ (git-ignored, holds the weights) to
results/training_logs/, so the per-epoch dev scores and the checkpoint choices are committed.

    python scripts/collect_training_logs.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT  # noqa: E402

EXP = ROOT / "exp"
OUT = ROOT / "results" / "training_logs"
FILES = {
    "gmm_hmm_dev_choice.json": EXP / "gmm_hmm" / "dev_choice.json",
    "e2e_ctc_train_info.json": EXP / "e2e_ctc" / "train_info.json",
    "e2e_aed_train_info.json": EXP / "e2e_aed" / "train_info.json",
    "e2e_hybrid_train_info.json": EXP / "e2e_hybrid" / "train_info.json",
    "e2e_rnnt_train_info.json": EXP / "e2e_rnnt" / "train_info.json",
    "whisper_small_ft_train_info.json": EXP / "whisper_small_ft" / "train_info.json",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in FILES.items():
        if src.exists():
            shutil.copyfile(src, OUT / name)
            print("copied", name)
        else:
            print("missing (not trained yet):", src.relative_to(ROOT))


if __name__ == "__main__":
    main()
