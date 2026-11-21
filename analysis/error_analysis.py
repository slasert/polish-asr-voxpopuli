"""Error analysis on the test split for every scored system.

Writes into analysis/:
  summary.md                    tables pasted into the README by scripts/make_tables.py
  substitutions_<system>.csv    top-20 substitution pairs with their error type
  worst_<system>.txt            jiwer alignment of the 10 worst clips (by word errors)
"""
from __future__ import annotations

import collections
import json
import sys
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score import load_predictions  # noqa: E402
from src.config import METRICS, ROOT  # noqa: E402
from src.eval.metrics import corpus_scores  # noqa: E402
from src.normalize import normalize  # noqa: E402

OUT = ROOT / "analysis"
STRIP = str.maketrans("ąćęłńóśźż", "acelnoszz")


def systems():
    return sorted(p.stem for p in METRICS.glob("*.json") if "test" in json.loads(p.read_text(encoding="utf-8")))


def sub_type(r, h):
    if r.translate(STRIP) == h.translate(STRIP):
        return "diacritic only"
    k = 0
    while k < min(len(r), len(h)) and r[k] == h[k]:
        k += 1
    if k >= 4 and k >= min(len(r), len(h)) - 3:
        return "inflection (same stem, different ending)"
    return "other"


def alignment_pairs(refs, hyps):
    out = jiwer.process_words(refs, hyps)
    subs, hits_by_ref = [], []
    for r, h, al in zip(out.references, out.hypotheses, out.alignments):
        hit = [False] * len(r)
        for c in al:
            if c.type == "substitute":
                subs += list(zip(r[c.ref_start_idx:c.ref_end_idx], h[c.hyp_start_idx:c.hyp_end_idx]))
            if c.type == "equal":
                for i in range(c.ref_start_idx, c.ref_end_idx):
                    hit[i] = True
        hits_by_ref.append((r, hit))
    return out, subs, hits_by_ref


def main():
    stems = yaml.safe_load(open(ROOT / "configs" / "domain_terms.yaml", encoding="utf-8"))["stems"]
    rows, sub_rows, bucket_rows, gender_rows, cost_rows = [], [], [], [], []
    for s in systems():
        m = json.loads((METRICS / f"{s}.json").read_text(encoding="utf-8"))["test"]
        df = load_predictions(s, "test").sort_values("utt_id").reset_index(drop=True)
        refs, hyps = list(df.text.map(normalize)), list(df.hyp.map(normalize))
        out, subs, hits = alignment_pairs(refs, hyps)
        n = m["normalized"]
        types = collections.Counter(sub_type(r, h) for r, h in subs)
        tot = max(1, len(subs))
        term_total = term_hit = 0
        for r, hit in hits:
            for w, ok in zip(r, hit):
                if any(w.startswith(st) for st in stems):
                    term_total += 1
                    term_hit += ok
        rows.append({"system": s, "S": n["sub"], "D": n["del"], "I": n["ins"],
                     "sub_diacritic": types["diacritic only"] / tot,
                     "sub_inflection": types["inflection (same stem, different ending)"] / tot,
                     "key_term_recall": term_hit / max(1, term_total), "key_terms": term_total})
        top = collections.Counter(subs).most_common(20)
        pd.DataFrame([{"ref": r, "hyp": h, "count": c, "type": sub_type(r, h)} for (r, h), c in top]).to_csv(
            OUT / f"substitutions_{s}.csv", index=False)
        # 10 worst clips
        err = [jiwer.process_words(r, h) for r, h in zip(refs, hyps)]
        nerr = np.array([e.substitutions + e.deletions + e.insertions for e in err])
        worst = np.argsort(-nerr, kind="stable")[:10]
        with open(OUT / f"worst_{s}.txt", "w", encoding="utf-8") as f:
            for i in worst:
                f.write(f"=== {df.utt_id[i]} ({nerr[i]} word errors)\n")
                f.write(jiwer.visualize_alignment(err[i], show_measures=False) + "\n")
        # buckets
        for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30)):
            sel = (df.duration.astype(float) >= lo) & (df.duration.astype(float) < hi)
            if sel.any():
                bucket_rows.append({"system": s, "bucket": f"{lo}-{hi} s",
                                    "wer": corpus_scores([refs[i] for i in df.index[sel]], [hyps[i] for i in df.index[sel]])["wer"]})
        for g in sorted(df.gender.unique()):
            sel = df.gender == g
            gender_rows.append({"system": s, "gender": g,
                                "wer": corpus_scores([refs[i] for i in df.index[sel]], [hyps[i] for i in df.index[sel]])["wer"]})
        cost_rows.append({"system": s, "rtfx": m["rtfx"], "peak_gpu_mb": m.get("peak_gpu_mb"),
                          "precision": m.get("precision", "fp32/fp16 autocast")})

    t = pd.DataFrame(rows)
    b = pd.DataFrame(bucket_rows).pivot(index="system", columns="bucket", values="wer")
    b = b[sorted(b.columns, key=lambda c: int(c.split("-")[0]))]  # 1-5 s, 5-10 s, ... not alphabetical
    g = pd.DataFrame(gender_rows).pivot(index="system", columns="gender", values="wer")
    c = pd.DataFrame(cost_rows).set_index("system")
    df = read_test_counts()
    pct = lambda x: f"{100 * x:.2f}"  # noqa: E731
    md = ["**Error types on test** (normalized text). *Inflection* = substitution where both words share "
          "a stem of at least 4 letters and differ only in the last ≤ 3 letters (a wrong case or verb ending); "
          "*diacritic only* = the words are equal once ą ć ę ł ń ó ś ź ż lose their diacritics. "
          "Key-term recall = share of reference words starting with one of the parliamentary stems in "
          "`configs/domain_terms.yaml` that the system got exactly right.", "",
          "| System | Substitutions | Deletions | Insertions | Inflection share of subs | Diacritic-only share of subs | Key-term recall (n) |",
          "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        md.append(f"| `{r.system}` | {r.S} | {r.D} | {r.I} | {pct(r.sub_inflection)} % | {pct(r.sub_diacritic)} % | "
                  f"{pct(r.key_term_recall)} % ({r.key_terms}) |")
    md += ["", f"**Test WER (%) by clip duration** (test clips per bucket: {df}):", "",
           "| System | " + " | ".join(b.columns) + " |", "|---" * (len(b.columns) + 1) + "|"]
    for sname, r in b.iterrows():
        md.append(f"| `{sname}` | " + " | ".join(pct(v) for v in r.values) + " |")
    md += ["", "**Test WER (%) by speaker gender:**", "", "| System | " + " | ".join(g.columns) + " |",
           "|---" * (len(g.columns) + 1) + "|"]
    for sname, r in g.iterrows():
        md.append(f"| `{sname}` | " + " | ".join(pct(v) for v in r.values) + " |")
    md += ["", "**Cost on test** (RTFx = audio seconds / compute seconds; RTX 3050 Laptop 4 GB):", "",
           "| System | RTFx | Peak GPU memory (MB) |", "|---|---|---|"]
    for sname, r in c.iterrows():
        md.append(f"| `{sname}` | {r.rtfx} | {r.peak_gpu_mb} |")
    md += finetune_vs_base("whisper_small_ft", "whisper_small")
    md += ["", "Per-system top-20 substitution pairs: `analysis/substitutions_<system>.csv`; alignments of the "
               "10 worst clips: `analysis/worst_<system>.txt`."]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def finetune_vs_base(ft, base):
    """Where the fine-tuned model gains or loses against its starting checkpoint, per split."""
    if not ((METRICS / f"{ft}.json").exists() and (METRICS / f"{base}.json").exists()):
        return []
    mf = json.loads((METRICS / f"{ft}.json").read_text(encoding="utf-8"))
    mb = json.loads((METRICS / f"{base}.json").read_text(encoding="utf-8"))
    md = ["", f"**Fine-tuned vs. base Whisper** (`{ft}` − `{base}`, normalized text; a clip is *worse* "
              "if it has more word errors after fine-tuning):", "",
          "| Split | WER base → fine-tuned | Δ substitutions | Δ deletions | Δ insertions | Clips worse | Clips better | Clips equal |",
          "|---|---|---|---|---|---|---|---|"]
    for split in ("dev", "test"):
        if split not in mf or split not in mb:
            continue
        a = load_predictions(base, split).sort_values("utt_id").reset_index(drop=True)
        b = load_predictions(ft, split).sort_values("utt_id").reset_index(drop=True)
        refs = list(a.text.map(normalize))
        diff = []
        for r, ha, hb in zip(refs, a.hyp.map(normalize), b.hyp.map(normalize)):
            ea, eb = jiwer.process_words(r, ha), jiwer.process_words(r, hb)
            diff.append((eb.substitutions + eb.deletions + eb.insertions)
                        - (ea.substitutions + ea.deletions + ea.insertions))
        diff = np.array(diff)
        nf, nb = mf[split]["normalized"], mb[split]["normalized"]
        md.append(f"| {split} | {100 * nb['wer']:.2f} → {100 * nf['wer']:.2f} | {nf['sub'] - nb['sub']:+d} | "
                  f"{nf['del'] - nb['del']:+d} | {nf['ins'] - nb['ins']:+d} | {(diff > 0).sum()} | "
                  f"{(diff < 0).sum()} | {(diff == 0).sum()} |")
    return md


def read_test_counts():
    from src.asr_io import read_split
    d = read_split("test").duration
    return ", ".join(f"{lo}-{hi} s: {int(((d >= lo) & (d < hi)).sum())}" for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30)))


if __name__ == "__main__":
    main()
