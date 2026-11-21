"""The single scorer for every system.

Reads  results/predictions/<system>/<split>.tsv   (utt_id <TAB> hyp)
       results/predictions/<system>/<split>.json  (run info written by the system's runner)
Writes results/metrics/<system>.json  (one entry per split; other splits are kept)

    python scripts/score.py --system whisper_small --split test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import METRICS, PRED, ROOT, SPLITS  # noqa: E402
from src.eval.metrics import corpus_scores  # noqa: E402
from src.normalize import normalize, orthographic  # noqa: E402

REQUIRED_RUN_KEYS = ["model_id", "revision", "decoding", "seed", "hardware",
                     "compute_seconds", "family", "trained_by_us"]


def read_tsv(p):
    return pd.read_csv(p, sep="\t", keep_default_na=False, quoting=3, dtype=str)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "uncommitted"


def load_predictions(system: str, split: str) -> pd.DataFrame:
    ref = read_tsv(SPLITS / f"{split}.tsv")
    hyp = read_tsv(PRED / system / f"{split}.tsv")
    if set(hyp.utt_id) != set(ref.utt_id) or not hyp.utt_id.is_unique:
        missing = set(ref.utt_id) - set(hyp.utt_id)
        raise SystemExit(f"{system}/{split}: predictions do not cover the split "
                         f"({len(missing)} missing, {len(hyp)} rows)")
    return ref.merge(hyp, on="utt_id", how="left")


def score(system: str, split: str) -> dict:
    df = load_predictions(system, split)
    run = json.loads((PRED / system / f"{split}.json").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_RUN_KEYS if k not in run]
    if missing:
        raise SystemExit(f"{system}/{split}.json lacks {missing}")
    norm = corpus_scores(df.text.map(normalize), df.hyp.map(normalize))
    orth = corpus_scores(df.text_orth.map(orthographic), df.hyp.map(orthographic))

    train = set(read_tsv(SPLITS / "train.tsv").text.map(normalize))
    unseen = df[~df.text.map(normalize).isin(train)]
    norm_unseen = corpus_scores(unseen.text.map(normalize), unseen.hyp.map(normalize))

    audio_s = df.duration.astype(float).sum()
    return {
        "system": system,
        "split": split,
        **{k: run[k] for k in run},
        "normalized": norm,
        "orthographic": {k: orth[k] for k in ("wer", "cer", "sub", "del", "ins", "n_ref_words")},
        "normalized_unseen_sentences": {k: norm_unseen[k] for k in ("wer", "cer", "n_utts")},
        "audio_seconds": round(audio_s, 3),
        "rtfx": round(audio_s / run["compute_seconds"], 2) if run["compute_seconds"] else None,
        "commit": git_commit(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args()
    out = METRICS / f"{a.system}.json"
    METRICS.mkdir(parents=True, exist_ok=True)
    allm = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    allm[a.split] = score(a.system, a.split)
    out.write_text(json.dumps(allm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    m = allm[a.split]
    print(f"{a.system} {a.split}: WER={m['normalized']['wer']:.4f} CER={m['normalized']['cer']:.4f} "
          f"orthWER={m['orthographic']['wer']:.4f} RTFx={m['rtfx']}")


if __name__ == "__main__":
    main()
