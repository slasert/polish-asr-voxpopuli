"""Pre-submission checks (repository hygiene + reproducibility). Exits 1 if anything fails.

    python scripts/check_submission.py [--members 5] [--rerun]
--rerun also re-runs 02_clean.py and 03_make_splits.py and checks nothing changed (slow).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLEANED, DATA, METRICS, RAW, ROOT, SPLITS  # noqa: E402

fails: list[str] = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.splitlines() if out.returncode == 0 else []


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    files = tracked()

    # contributions/
    cdir = ROOT / "contributions"
    cfiles = sorted(cdir.glob("*.md")) if cdir.exists() else []
    check(cdir.exists(), "contributions/ exists")
    bad = [f.name for f in cfiles if not re.match(r"^[^_]+_[0-9]+_[A-Za-z0-9-]+\.md$", f.name)]
    check(not bad, f"contributions file names match NAME_ID_USERNAME.md {bad or ''}")
    ph = [f.name for f in cfiles if re.search(r"PENDING|TODO|USERNAME", f.read_text(encoding="utf-8"))]
    check(not ph, f"no placeholders in contributions files {ph or ''}")
    check(len(cfiles) == a.members, f"{len(cfiles)} contributions files for {a.members} members")

    # authors
    out = subprocess.run(["git", "shortlog", "-sne", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    authors = {re.sub(r"^\s*\d+\s+", "", l) for l in out.splitlines() if l.strip()}
    check(len(authors) >= a.members, f"{len(authors)} distinct commit authors (need {a.members})")

    check((ROOT / "LICENSE").exists(), "LICENSE at repo root")

    stray = [f for f in files if re.search(r"__pycache__|\.pyc$|\.ipynb_checkpoints|\.(pt|ckpt|safetensors|bin)$", f)]
    check(not stray, f"no caches or weights tracked {stray[:5] or ''}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    roots = sorted({f.split("/")[0] for f in files} or {p.name for p in ROOT.iterdir() if p.name != ".git"})
    missing = [r for r in roots if f"`{r}" not in readme]
    check(not missing, f"every root file/folder explained in README {missing or ''}")

    unpinned = []
    for req in ROOT.glob("requirements*.txt"):
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-") and "==" not in line:
                unpinned.append(f"{req.name}: {line}")
    check(not unpinned, f"all requirements pinned {unpinned[:5] or ''}")

    check(any((RAW / "audio").glob("*.wav")), "data/raw has audio")
    check(any((CLEANED / "audio").glob("*.flac")), "data/cleaned has audio")
    check((DATA / "cleaning_log.csv").exists(), "data/cleaning_log.csv exists")

    import pandas as pd
    spk = {s: set(pd.read_csv(SPLITS / f"{s}.tsv", sep="\t", quoting=3, dtype=str).speaker) for s in ("train", "dev", "test")}
    check(not (spk["train"] & spk["dev"] or spk["train"] & spk["test"] or spk["dev"] & spk["test"]), "no speaker overlap between splits")

    if a.rerun:
        watched = list(CLEANED.rglob("*")) + list(SPLITS.glob("*.tsv")) + [DATA / "cleaning_log.csv"]
        watched = [p for p in watched if p.is_file()]
        before = digest(watched)
        for s in ("02_clean.py", "03_make_splits.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT, check=True, capture_output=True)
        check(digest([p for p in list(CLEANED.rglob("*")) + list(SPLITS.glob("*.tsv")) + [DATA / "cleaning_log.csv"] if p.is_file()]) == before,
              "re-running cleaning and splitting reproduces the committed files")

    for p in sorted(METRICS.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        for split, e in m.items():
            need = ["model_id", "seed", "commit"] + ([] if e.get("trained_by_us") else ["revision"])
            lack = [k for k in need if k not in e]
            check(not lack, f"{p.name}[{split}] has {need} {lack or ''}")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    check(r.returncode == 0, "README numbers match results/metrics (make_tables.py --check)")

    for nb in ROOT.rglob("*.ipynb"):
        txt = nb.read_text(encoding="utf-8")
        check(not re.search(r"files\.upload\(|drive\.mount\(", txt), f"{nb.name} has no Colab uploads/Drive mounts")

    print(f"\n{len(fails)} check(s) failed" if fails else "\nall checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
