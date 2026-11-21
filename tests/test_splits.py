"""Split tests: speaker/recording disjointness, coverage, and seed determinism."""
import importlib.util

import pandas as pd

from src.config import CLEANED, ROOT, SPLITS, seed

spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "03_make_splits.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def read(p):
    return pd.read_csv(p, sep="\t", keep_default_na=False, quoting=3, dtype=str)


PARTS = {s: read(SPLITS / f"{s}.tsv") for s in ("train", "dev", "test")}


def test_speakers_disjoint():
    s = {k: set(v.speaker) for k, v in PARTS.items()}
    assert not (s["train"] & s["dev"]) and not (s["train"] & s["test"]) and not (s["dev"] & s["test"])


def test_recordings_disjoint():
    # session recording = the part of the VoxPopuli id before the segment index
    rec = {k: set(v.utt_id.str.rsplit("_", n=1).str[0]) for k, v in PARTS.items()}
    assert not (rec["train"] & rec["dev"]) and not (rec["train"] & rec["test"]) and not (rec["dev"] & rec["test"])


def test_every_cleaned_clip_in_exactly_one_split():
    cleaned = read(CLEANED / "metadata.tsv")
    ids = pd.concat([v.utt_id for v in PARTS.values()])
    assert ids.is_unique and set(ids) == set(cleaned.utt_id)


def test_same_seed_same_split():
    meta = read(CLEANED / "metadata.tsv")
    meta["duration"] = meta.duration.astype(float)
    a, b = mk.assign(meta, seed()), mk.assign(meta, seed())
    assert a == b
    on_disk = {r.speaker: k for k, v in PARTS.items() for r in v.itertuples()}
    assert a == on_disk


def test_dev_test_hours_close_to_target():
    hours = {k: v.duration.astype(float).sum() / 3600 for k, v in PARTS.items()}
    total = sum(hours.values())
    for k in ("dev", "test"):
        assert 0.07 <= hours[k] / total <= 0.13
