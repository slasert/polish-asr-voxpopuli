"""Build every results table from results/metrics/*.json (and the data logs), write them to
results/tables/, and paste them into README.md between <!-- BEGIN:<name> --> / <!-- END:<name> -->.

    python scripts/make_tables.py           # regenerate tables and README blocks
    python scripts/make_tables.py --check   # fail if README differs from the generated tables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, METRICS, ROOT, TABLES  # noqa: E402

README = ROOT / "README.md"
REPO_URL = "https://huggingface.co/"
# systems whose test numbers belong in the tables, in display order
PRETRAINED = ["wav2vec2_xlsr53_pl", "hubert_large_pl_s6", "hubert_large_pl_s484", "hubert_large_pl_s995",
              "whisper_small", "canary_1b_v2", "parakeet_tdt_0.6b_v3", "qwen3_asr_0.6b"]
TRAINED = ["gmm_hmm", "e2e_ctc", "e2e_aed", "e2e_hybrid", "e2e_rnnt"]
CONTRIB = ["whisper_small_ft"]


def pct(x):
    return f"{100 * x:.2f}"


def load(system):
    p = METRICS / f"{system}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def link(model_id, rev):
    if model_id.startswith("ours"):
        return model_id
    return f"[`{model_id}`]({REPO_URL}{model_id}/tree/{rev}) @ `{rev[:10]}`"


def table_a():
    rows = ["| System | Family | Exact checkpoint @ revision | Language setting | Decoding | Dev WER | Test WER (norm) | Test CER (norm) | Test WER (orth) | Test CER (orth) | RTFx |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in PRETRAINED:
        m = load(s)
        if "test" not in m:
            continue
        t, d = m["test"], m.get("dev", {})
        rows.append(f"| `{s}` | {t['family'].split(' (')[0]} | {link(t['model_id'], t['revision'])} | {t['language']} | "
                    f"{t['decoding']} | {pct(d['normalized']['wer']) if d else '-'} | **{pct(t['normalized']['wer'])}** | "
                    f"{pct(t['normalized']['cer'])} | {pct(t['orthographic']['wer'])} | {pct(t['orthographic']['cer'])} | {t['rtfx']} |")
    return "\n".join(rows)


def table_hubert_dev():
    rows = ["| HuBERT checkpoint (B6 candidates) | Dev WER | Dev CER | Used on test |", "|---|---|---|---|"]
    cands = [s for s in PRETRAINED if s.startswith("hubert") and "dev" in load(s)]
    if not cands:
        return ""
    best = min(cands, key=lambda s: load(s)["dev"]["normalized"]["wer"])
    for s in cands:
        d = load(s)["dev"]
        rows.append(f"| {link(d['model_id'], d['revision'])} | {pct(d['normalized']['wer'])} | "
                    f"{pct(d['normalized']['cer'])} | {'yes (lowest dev WER)' if s == best else 'no'} |")
    return "\n".join(rows)


def table_b():
    rows = ["| System | Family | Toolkit / recipe | Features | Train hours | Selected (on dev) | Decoding | Dev WER | Test WER | Test CER | Test WER (orth) | RTFx |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in TRAINED:
        m = load(s)
        if "test" not in m:
            continue
        t, d = m["test"], m.get("dev", {})
        sel = f"epoch {t['selected_epoch']} of {t['epochs']}" if "selected_epoch" in t else "LM weight + insertion bonuses (grid)"
        rows.append(f"| `{s}` | {t['family']} | {t['toolkit']} | {t['features']} | {t['train_hours']} | {sel} | {t['decoding']} | "
                    f"{pct(d['normalized']['wer']) if d else '-'} | **{pct(t['normalized']['wer'])}** | {pct(t['normalized']['cer'])} | "
                    f"{pct(t['orthographic']['wer'])} | {t['rtfx']} |")
    return "\n".join(rows)


def table_c():
    rows = ["| System | Base checkpoint @ revision | What we did | Train hours | Steps | Dev WER | Test WER | Test CER | Test WER (orth) | ΔWER vs. base (95% CI) | ΔCER vs. base (95% CI) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in CONTRIB:
        m = load(s)
        if "test" not in m:
            continue
        t, d = m["test"], m.get("dev", {})
        cfg = t["config"]
        bs = METRICS / "bootstrap" / f"{s}__vs__whisper_small__test.json"
        b = json.loads(bs.read_text(encoding="utf-8")) if bs.exists() else None
        fmt = (lambda k: f"{pct(b[k]['delta'])} [{pct(b[k]['ci95'][0])}, {pct(b[k]['ci95'][1])}]") if b else (lambda k: "-")
        rows.append(f"| `{s}` | {link(cfg['base_model'], cfg['base_revision'])} | full fine-tuning, {cfg['optim']}, "
                    f"lr {cfg['learning_rate']}, effective batch {cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']}, "
                    f"early stopping on dev WER | {t['train_hours']} | {t['steps']} | {pct(d['normalized']['wer']) if d else '-'} | "
                    f"**{pct(t['normalized']['wer'])}** | {pct(t['normalized']['cer'])} | {pct(t['orthographic']['wer'])} | {fmt('wer')} | {fmt('cer')} |")
    return "\n".join(rows)


def table_bootstrap():
    rows = ["| A | B | Split | ΔWER (A−B) | 95% CI | ΔCER (A−B) | 95% CI | Resamples |", "|---|---|---|---|---|---|---|---|"]
    for p in sorted((METRICS / "bootstrap").glob("*.json")):
        b = json.loads(p.read_text(encoding="utf-8"))
        rows.append(f"| `{b['a']}` | `{b['b']}` | {b['split']} | {pct(b['wer']['delta'])} | [{pct(b['wer']['ci95'][0])}, "
                    f"{pct(b['wer']['ci95'][1])}] | {pct(b['cer']['delta'])} | [{pct(b['cer']['ci95'][0])}, {pct(b['cer']['ci95'][1])}] | {b['resamples']} |")
    return "\n".join(rows)


def table_unseen():
    rows = ["| System | Test WER (all) | Test WER (sentences not in train) | Utterances not in train |", "|---|---|---|---|"]
    for s in PRETRAINED + TRAINED + CONTRIB:
        t = load(s).get("test")
        if t:
            u = t["normalized_unseen_sentences"]
            rows.append(f"| `{s}` | {pct(t['normalized']['wer'])} | {pct(u['wer'])} | {u['n_utts']} |")
    return "\n".join(rows)


def table_cleaning():
    log = pd.read_csv(DATA / "cleaning_log.csv", keep_default_na=False)
    raw = pd.read_csv(DATA / "raw" / "metadata.tsv", sep="\t", keep_default_na=False, quoting=3)
    import soundfile as sf
    dur = {Path(f).stem: sf.info(DATA / "raw" / f).duration for f in raw.file}
    log["hours"] = log.utt_id.map(dur) / 3600
    g = log.groupby(["issue", "action"]).agg(n=("utt_id", "size"), hours=("hours", "sum")).reset_index()
    rows = ["| Reason code | Action | Clips | Hours |", "|---|---|---|---|"]
    for r in g.sort_values(["action", "n"], ascending=[False, False]).itertuples():
        rows.append(f"| `{r.issue}` | {r.action} | {r.n} | {r.hours:.3f} |")
    rem = log[log.action == "removed"]
    rows.append(f"| **total removed** | | **{len(rem)}** of {len(raw)} | **{rem.hours.sum():.3f}** of {sum(dur.values()) / 3600:.2f} |")
    return "\n".join(rows)


def blocks():
    b = {"cleaning": table_cleaning(), "split_stats": (TABLES / "split_stats.md").read_text(encoding="utf-8").split("\n", 2)[2].strip(),
         "table_a": table_a(), "hubert_dev": table_hubert_dev(), "table_b": table_b(), "table_c": table_c(),
         "bootstrap": table_bootstrap(), "unseen": table_unseen()}
    audit = TABLES / "audit.md"
    b["audit"] = (audit.read_text(encoding="utf-8").strip() if audit.exists() else
                  "**Manual audit:** not done yet. `scripts/05_audit.py sample` drew 100 random cleaned clips "
                  "(`data/audit/audit_sample.tsv`); a team member listens to them in `data/audit/listen.html`, "
                  "saves `data/audit/audit_results.csv` and runs `scripts/05_audit.py report`.")
    analysis = ROOT / "analysis" / "summary.md"
    if analysis.exists():
        b["analysis"] = analysis.read_text(encoding="utf-8").strip()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bl = blocks()
    TABLES.mkdir(parents=True, exist_ok=True)
    readme = README.read_text(encoding="utf-8")
    new = readme
    for name, content in bl.items():
        # the block may be empty (BEGIN directly followed by END)
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        if not pat.search(new):
            sys.exit(f"README has no block {name}")
        new = pat.sub(lambda m: m.group(1) + content + "\n" + m.group(2), new)
    if a.check:
        if new != readme:
            sys.exit("README tables differ from results/ (run scripts/make_tables.py)")
        print("README tables match results/")
        return
    for name, content in bl.items():
        if name not in ("split_stats", "analysis", "audit"):  # those files are owned by other scripts
            (TABLES / f"{name}.md").write_text(content + "\n", encoding="utf-8")
    README.write_text(new, encoding="utf-8")
    print("tables written:", ", ".join(bl))


if __name__ == "__main__":
    main()
