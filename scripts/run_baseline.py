"""One entry point for the pretrained baselines (B5-B10).

    python scripts/run_baseline.py --system whisper_small --split dev
    python scripts/score.py        --system whisper_small --split dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.pretrained.systems import RUNNERS  # noqa: E402
from src.asr_io import read_split, write_predictions  # noqa: E402
from src.config import set_all_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    s = set_all_seeds()
    df = read_split(a.split)
    ids, hyps, run = RUNNERS[a.system](a.system, df)
    run.update(seed=s, trained_by_us=False)
    out = write_predictions(a.system, a.split, ids, hyps, run)
    print(f"wrote {out}/{a.split}.tsv ({len(ids)} utts, {run['compute_seconds']} s)")


if __name__ == "__main__":
    main()
