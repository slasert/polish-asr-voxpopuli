"""Speaker-disjoint train/dev/test split of data/cleaned/ (80/10/10 by hours).

- The unit of assignment is the speaker: all clips of a speaker go to one split.
  (Every VoxPopuli clip is a segment of one speaker's speech in one plenary session, so a
  speaker group also keeps every recording in one split.)
- Speakers are shuffled with the global seed, separately for each gender, and filled
  greedily into dev, then test, then train, so dev and test get roughly the corpus gender mix.
- A speaker with more audio than half the dev target goes to train (otherwise one speaker
  could make up most of dev or test).
- We ignore VoxPopuli's own validation/test labels: 35 speakers appear in both of them
  (see results/tables/split_stats.md), so they are not speaker-disjoint.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLEANED, SPLITS, seed  # noqa: E402

FRACTIONS = {"dev": 0.10, "test": 0.10}


def assign(meta: pd.DataFrame, s: int) -> dict[str, str]:
    spk = meta.groupby("speaker").agg(hours=("duration", lambda d: d.sum() / 3600),
                                      gender=("gender", "first")).reset_index()
    total = spk.hours.sum()
    rng = random.Random(s)
    out = {}
    for gender, g in sorted(spk.groupby("gender"), key=lambda x: x[0]):
        share = g.hours.sum() / total
        targets = {k: v * total * share for k, v in FRACTIONS.items()}
        filled = {k: 0.0 for k in FRACTIONS}
        rows = sorted(g.itertuples(index=False), key=lambda r: r.speaker)
        rng.shuffle(rows)
        for r in rows:
            split = "train"
            if r.hours <= 0.5 * targets["dev"]:
                for k in ("dev", "test"):
                    if filled[k] + r.hours <= targets[k] * 1.05 or filled[k] == 0:
                        split = k
                        filled[k] += r.hours
                        break
            out[r.speaker] = split
    return out


def main():
    meta = pd.read_csv(CLEANED / "metadata.tsv", sep="\t", keep_default_na=False, quoting=3,
                       dtype={"speaker": str})
    meta["duration"] = meta["duration"].astype(float)
    mapping = assign(meta, seed())
    meta["split"] = meta["speaker"].map(mapping)
    SPLITS.mkdir(parents=True, exist_ok=True)
    cols = ["utt_id", "path", "speaker", "gender", "duration", "text", "text_orth"]
    meta["path"] = "data/cleaned/" + meta["file"]
    for split in ("train", "dev", "test"):
        part = meta[meta.split == split].sort_values("utt_id")
        part[cols].to_csv(SPLITS / f"{split}.tsv", sep="\t", index=False, quoting=3,
                          lineterminator="\n")
        print(f"{split}: {len(part)} utts, {part.speaker.nunique()} speakers, "
              f"{part.duration.sum() / 3600:.2f} h")


if __name__ == "__main__":
    main()
