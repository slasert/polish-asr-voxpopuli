"""Shared input/output for every ASR system: read a split, load audio, write predictions."""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.config import PRED, ROOT, SPLITS


def read_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS / f"{split}.tsv", sep="\t", keep_default_na=False, quoting=3,
                     dtype={"speaker": str})
    df["duration"] = df["duration"].astype(float)
    return df


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(ROOT / path, dtype="float32")
    assert sr == 16000 and x.ndim == 1, path
    return x


def hardware() -> str:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        return f"{gpu}; torch {torch.__version__}; {platform.system()} {platform.release()}"
    except ImportError:
        return platform.platform()


@contextmanager
def timer():
    """Wall-clock seconds spent inside the block (model loading excluded by the caller)."""
    t = {"s": 0.0}
    start = time.perf_counter()
    try:
        yield t
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        t["s"] = time.perf_counter() - start


def peak_gpu_mb() -> float | None:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
    except ImportError:
        return None


def write_predictions(system: str, split: str, ids, hyps, run: dict) -> Path:
    out = PRED / system
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(zip(ids, hyps))
    with open(out / f"{split}.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("utt_id\thyp\n")
        for i, h in rows:
            h = " ".join(str(h).replace("\t", " ").split())
            f.write(f"{i}\t{h}\n")
    run = {**run, "hardware": run.get("hardware", hardware()), "peak_gpu_mb": peak_gpu_mb()}
    (out / f"{split}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return out
