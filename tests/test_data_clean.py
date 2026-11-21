"""Data-pipeline tests on the committed raw and cleaned data."""
import hashlib
import subprocess
import sys

import pandas as pd
import pytest
import soundfile as sf

from src.config import CLEANED, DATA, RAW, ROOT


def read_tsv(p, **kw):
    return pd.read_csv(p, sep="\t", keep_default_na=False, quoting=3, dtype=str, **kw)


@pytest.fixture(scope="module")
def cleaned():
    return read_tsv(CLEANED / "metadata.tsv")


@pytest.fixture(scope="module")
def raw():
    return read_tsv(RAW / "metadata.tsv")


@pytest.fixture(scope="module")
def log():
    return pd.read_csv(DATA / "cleaning_log.csv", keep_default_na=False, dtype=str)


def test_raw_checksums_match(raw):
    for r in raw.sample(200, random_state=0).itertuples():
        assert hashlib.sha256((RAW / r.file).read_bytes()).hexdigest() == r.sha256


def test_every_cleaned_clip_is_16k_mono_nonempty(cleaned):
    for f in cleaned.file:
        info = sf.info(CLEANED / f)
        assert info.samplerate == 16000 and info.channels == 1 and info.frames > 0


def test_no_empty_transcripts(cleaned):
    assert (cleaned.text.str.strip() != "").all()
    assert (cleaned.text_orth.str.strip() != "").all()


def test_no_duplicate_audio(cleaned):
    hashes = [hashlib.sha256(sf.read(CLEANED / f, dtype="int16")[0].tobytes()).hexdigest()
              for f in cleaned.file]
    assert len(hashes) == len(set(hashes))


def test_every_removed_clip_is_logged(raw, cleaned, log):
    raw_ids = set(raw.file.str.replace("audio/", "").str.replace(".wav", ""))
    removed = raw_ids - set(cleaned.utt_id)
    logged = set(log[log.action == "removed"].utt_id)
    assert removed == logged
    assert not (set(cleaned.utt_id) & logged)


def test_log_reasons_are_known(log):
    known = {"CORRUPT_AUDIO", "LOW_VALIDATION", "EMPTY_TRANSCRIPT", "MISSING_ORTH",
             "NUM_UNVERIFIED", "CHARSET_RESTORED", "UNICODE_FIX", "BAD_CHARSET",
             "TEXT_MISMATCH", "DURATION_OUTLIER", "SILENCE", "CLIPPED", "RATE_MISALIGN",
             "DUP_AUDIO", "DUP_SEGMENT", "DUP_UTT"}
    assert set(log.issue) <= known


def test_durations_within_thresholds(cleaned):
    d = cleaned.duration.astype(float)
    assert d.min() >= 1.0 and d.max() <= 30.0


def test_cleaned_text_has_no_digits(cleaned):
    assert not cleaned.text.str.contains(r"\d").any()


def test_foreign_letters_restored(cleaned):
    assert cleaned.text.str.contains("pöttering").any()
    assert not cleaned.text.str.contains(r"\bpttering\b").any()


def _digest():
    h = hashlib.sha256()
    for p in sorted(CLEANED.rglob("*")) + [DATA / "cleaning_log.csv"]:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.slow
def test_cleaning_is_deterministic():
    before = _digest()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "02_clean.py")], check=True)
    assert _digest() == before
