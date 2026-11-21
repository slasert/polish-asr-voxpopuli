"""Manual label audit of data/cleaned/: a person listens to 100 random cleaned clips and marks
whether the transcript matches the audio. This script never judges a clip itself.

    python scripts/05_audit.py sample   # draw the clips -> data/audit/audit_sample.tsv + listen.html
    (open data/audit/listen.html in Chrome/Edge, listen, mark, "Download CSV",
     save the file as data/audit/audit_results.csv)
    python scripts/05_audit.py report   # -> results/tables/audit.md (pasted into README by make_tables.py)

The sample is drawn from all cleaned clips with the global seed, so it is the same on every run.
A clip counts as a label error if any word of `text_orth` differs from what is said
(wrong, missing or extra word, or text that belongs to other audio). Punctuation and casing are
not judged: every metric is computed on normalized text.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLEANED, DATA, TABLES, seed  # noqa: E402

AUDIT = DATA / "audit"
N = 100
VERDICTS = {"ok": "transcript matches the audio",
            "wrong_word": "wrong word(s)",
            "missing_word": "spoken word(s) missing from the transcript",
            "extra_word": "transcript has word(s) that are not spoken",
            "mismatch": "transcript belongs to other audio / mostly wrong",
            "unclear": "audio too unclear to judge"}


def sample():
    meta = pd.read_csv(CLEANED / "metadata.tsv", sep="\t", keep_default_na=False, quoting=3,
                       dtype={"speaker": str})
    s = meta.sample(n=N, random_state=seed()).sort_values("utt_id").reset_index(drop=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    s[["utt_id", "file", "speaker", "gender", "duration", "text_orth"]].to_csv(
        AUDIT / "audit_sample.tsv", sep="\t", index=False, quoting=3, lineterminator="\n")
    clips = [{"id": r.utt_id, "src": f"../cleaned/{r.file}", "dur": r.duration, "text": r.text_orth}
             for r in s.itertuples()]
    page = PAGE.replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)).replace(
        "__VERDICTS__", json.dumps(VERDICTS))
    (AUDIT / "listen.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"{N} clips, {s.duration.sum() / 60:.1f} min of audio -> {AUDIT / 'listen.html'}")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def report():
    want = pd.read_csv(AUDIT / "audit_sample.tsv", sep="\t", keep_default_na=False, quoting=3)
    res = pd.read_csv(AUDIT / "audit_results.csv", keep_default_na=False)
    assert set(res.utt_id) == set(want.utt_id), "audit_results.csv does not cover exactly the sampled clips"
    bad = set(res.verdict) - set(VERDICTS)
    assert not bad and (res.verdict != "").all(), f"unmarked clips or unknown verdicts: {bad}"
    judged = res[res.verdict != "unclear"]
    k, n = int((judged.verdict != "ok").sum()), len(judged)
    lo, hi = wilson(k, n)
    lines = [f"**Manual audit** (`scripts/05_audit.py`): {res.auditor.iloc[0]} listened to {len(res)} random "
             f"cleaned clips (seed {seed()}) and compared each with `text_orth` (punctuation and casing "
             f"not judged). **Label error rate: {100 * k / n:.1f} %** ({k} of {n} judged clips; "
             f"95% Wilson CI [{100 * lo:.1f}, {100 * hi:.1f}] %).", "",
             "| Verdict | Clips |", "|---|---|"]
    for v, desc in VERDICTS.items():
        lines.append(f"| {desc} | {int((res.verdict == v).sum())} |")
    notes = res[~res.verdict.isin(["ok", "unclear"]) & (res.note != "")]
    if len(notes):
        lines += ["", "Errors found (clip: note):", ""] + [f"- `{r.utt_id}`: {r.note}" for r in notes.itertuples()]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label audit</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#fff;color:#111}
.clip{border:1px solid #ccc;border-radius:6px;padding:10px;margin:10px 0}
.clip.done{background:#f1f8f1}.clip.err{background:#fdf1f1}
.text{font-size:1.1em;margin:8px 0}.meta{color:#666;font-size:.85em}
audio{width:100%}select,input[type=text]{font-size:1em;margin-right:8px}input[type=text]{width:50%}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc}
</style></head><body>
<div id="bar"><b>Label audit</b> &mdash; listen to each clip and choose a verdict.
Marked: <span id="count">0</span>/<span id="total"></span>.
Your name: <input type="text" id="auditor" style="width:12em">
<button id="dl">Download CSV</button>
<div class="meta">Answers are kept in this browser while you work. Save the CSV as data/audit/audit_results.csv.</div></div>
<div id="list"></div>
<script>
const CLIPS = __CLIPS__, VERDICTS = __VERDICTS__, KEY = "aek-audit-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const list = document.getElementById("list");
document.getElementById("total").textContent = CLIPS.length;
const auditor = document.getElementById("auditor");
auditor.value = state._auditor || "";
auditor.oninput = () => { state._auditor = auditor.value; save(); };
function refresh() {
  let n = 0;
  CLIPS.forEach((c, i) => {
    const s = state[c.id] || {}, el = document.getElementById("c" + i);
    el.className = "clip" + (s.v ? (s.v === "ok" ? " done" : " err") : "");
    if (s.v) n++;
  });
  document.getElementById("count").textContent = n;
}
CLIPS.forEach((c, i) => {
  const d = document.createElement("div");
  d.id = "c" + i;
  const opts = ['<option value="">-- verdict --</option>'].concat(
    Object.entries(VERDICTS).map(([k, v]) => `<option value="${k}">${v}</option>`)).join("");
  d.innerHTML = `<div class="meta">${i + 1}. ${c.id} (${c.dur} s)</div>
    <audio controls preload="none" src="${c.src}"></audio>
    <div class="text"></div>
    <select>${opts}</select><input type="text" placeholder="note: what is wrong (optional)">`;
  d.querySelector(".text").textContent = c.text;
  const sel = d.querySelector("select"), note = d.querySelector("input");
  const s = state[c.id] || {};
  sel.value = s.v || ""; note.value = s.n || "";
  sel.onchange = () => { state[c.id] = {...(state[c.id] || {}), v: sel.value}; save(); refresh(); };
  note.oninput = () => { state[c.id] = {...(state[c.id] || {}), n: note.value}; save(); };
  list.appendChild(d);
});
refresh();
document.getElementById("dl").onclick = () => {
  const missing = CLIPS.filter(c => !(state[c.id] || {}).v).length;
  if (missing && !confirm(missing + " clips have no verdict yet. Download anyway?")) return;
  if (!auditor.value.trim()) { alert("Please enter your name first."); return; }
  const q = x => '"' + String(x || "").replace(/"/g, '""') + '"';
  const rows = ["utt_id,verdict,note,auditor"].concat(CLIPS.map(c => {
    const s = state[c.id] || {};
    return [c.id, s.v, s.n, auditor.value.trim()].map(q).join(",");
  }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  a.download = "audit_results.csv";
  a.click();
};
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "report"])
    {"sample": sample, "report": report}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
