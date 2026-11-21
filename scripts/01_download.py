"""Provenance of data/raw/. The raw data is already committed; this script shows
exactly how it was produced and can rebuild it.

Source: Hugging Face dataset facebook/voxpopuli, config "pl", pinned revision.
We take the official `validation` and `test` parquet files as our pool and write
every clip's audio bytes to disk *unchanged* (no decoding, no resampling).

    python scripts/01_download.py --cache /path/to/parquet_cache
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW  # noqa: E402

REPO = "facebook/voxpopuli"
REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
FILES = {"validation": "pl/validation-00000-of-00001.parquet",
         "test": "pl/test-00000-of-00001.parquet"}
COLUMNS = ["audio_id", "source_split", "file", "speaker_id", "gender", "accent",
           "is_gold_transcript", "raw_text", "normalized_text", "sha256"]


def safe_name(audio_id: str) -> str:
    # ':' is not allowed in Windows file names; the original id is kept in metadata.tsv
    return audio_id.replace(":", "-") + ".wav"


def tsv_field(s) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder for the downloaded parquet files")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    audio_dir = RAW / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, path in FILES.items():
        local = cache / f"pl_{split}.parquet"
        if not local.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
            print("downloading", url)
            urllib.request.urlretrieve(url, local)
        table = pq.read_table(local).to_pylist()
        for r in table:
            data = r["audio"]["bytes"]
            name = safe_name(r["audio_id"])
            (audio_dir / name).write_bytes(data)
            rows.append([r["audio_id"], split, f"audio/{name}", r["speaker_id"], r["gender"],
                         r["accent"], r["is_gold_transcript"], r["raw_text"],
                         r["normalized_text"], hashlib.sha256(data).hexdigest()])

    rows.sort(key=lambda x: x[0])
    with open(RAW / "metadata.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(tsv_field(x) for x in r) + "\n")
    with open(RAW / "CHECKSUMS.sha256", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(f"{r[-1]}  {r[2]}\n")
    print(f"wrote {len(rows)} clips to {audio_dir}")


if __name__ == "__main__":
    main()
