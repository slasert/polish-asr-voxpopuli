"""Paired bootstrap of the corpus-WER/CER difference between two systems on the same split.

Each resample draws utterances with replacement (same draw for both systems), re-pools
errors and reference words, and records delta = WER(A) - WER(B). We report the 95%
percentile interval and the fraction of resamples where A is not better (p-value style).

    python scripts/bootstrap.py --a whisper_small_ft --b whisper_small --split test
Writes results/metrics/bootstrap/<a>__vs__<b>__<split>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score import load_predictions  # noqa: E402
from src.config import METRICS, seed  # noqa: E402
from src.eval.metrics import per_utt_errors  # noqa: E402
from src.normalize import normalize  # noqa: E402


def errors(system, split):
    df = load_predictions(system, split).sort_values("utt_id")
    return np.array(per_utt_errors(df.text.map(normalize), df.hyp.map(normalize)), dtype=float)


def paired_bootstrap(ea, eb, n, rng):
    assert ea.shape == eb.shape
    k = len(ea)
    idx = rng.integers(0, k, size=(n, k))
    out = {}
    for name, (e, r) in {"wer": (0, 1), "cer": (2, 3)}.items():
        ref = ea[:, r][idx].sum(1)
        da = ea[:, e][idx].sum(1) / ref
        db = eb[:, e][idx].sum(1) / ref
        d = da - db
        point = ea[:, e].sum() / ea[:, r].sum() - eb[:, e].sum() / eb[:, r].sum()
        out[name] = {"delta": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "p_a_not_better": float((d >= 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    s = seed()
    res = paired_bootstrap(errors(a.a, a.split), errors(a.b, a.split), a.n, np.random.default_rng(s))
    res.update(a=a.a, b=a.b, split=a.split, resamples=a.n, seed=s)
    out = METRICS / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.a}__vs__{a.b}__{a.split}.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
