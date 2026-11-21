"""Shared paths and the global seed."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEANED = DATA / "cleaned"
SPLITS = DATA / "splits"
RESULTS = ROOT / "results"
PRED = RESULTS / "predictions"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"


def seed() -> int:
    with open(ROOT / "configs" / "seeds.yaml", encoding="utf-8") as f:
        return int(yaml.safe_load(f)["seed"])


def set_all_seeds(s: int | None = None) -> int:
    s = seed() if s is None else s
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass
    return s
