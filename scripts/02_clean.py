"""Clean data/raw/ -> data/cleaned/ and log every removal or fix to data/cleaning_log.csv.

Checks (first failing removal check decides the reason; fixes are logged separately):
  CORRUPT_AUDIO        unreadable file, zero length, wrong sample rate or channel count
  LOW_VALIDATION       VoxPopuli is_gold_transcript is False
  EMPTY_TRANSCRIPT     empty normalized transcript
  MISSING_ORTH         empty raw_text, so no orthographic reference exists
  NUM_UNVERIFIED       raw_text has digits: the spelled-out reference was generated
                       automatically by VoxPopuli and is often wrongly inflected or misspelled
  CHARSET_RESTORED     (fix) letters outside the Polish alphabet that VoxPopuli's normalizer
                       deleted ("Pöttering" -> "pttering") are restored from raw_text
  UNICODE_FIX          (fix) transcript was not NFC
  BAD_CHARSET          digits or non-Latin characters left in the transcript
  TEXT_MISMATCH        raw_text and normalized_text disagree on words (after the fixes above)
  DURATION_OUTLIER     shorter than min or longer than max duration
  SILENCE              mostly silent frames, or the whole clip is near-silent
  CLIPPED              too many samples at full scale
  RATE_MISALIGN        characters per second far outside the corpus distribution
  DUP_AUDIO            identical audio bytes (keep one)
  DUP_SEGMENT          same plenary segment (same date-time-index) published under two session
                       ids; keep the longest copy
  DUP_UTT              same (speaker, transcript) pair (keep one)

Resampling/format conversion is not cleaning: we write 16-bit FLAC (lossless here, because the
raw 32-bit float samples are exact multiples of 1/32768; the script asserts this).
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLEANED, DATA, RAW, ROOT  # noqa: E402
from src.normalize import normalize  # noqa: E402

SR = 16000
POLISH_LETTERS = set("aąbcćdeęfghijklłmnńoóprsśtuwyzźż") | set("qvx")
LOG_COLUMNS = ["utt_id", "speaker", "issue", "action", "detail"]


def words(s: str) -> list[str]:
    # U+02EE (modifier letter double apostrophe) is used as a closing quote in raw_text
    return re.sub(r"[^\w\s']|_|ˮ", " ", s.lower()).split()


def restore_foreign(raw: str, norm: str) -> tuple[str, list[tuple[str, str]]]:
    """Put back letters that VoxPopuli's normalizer deleted, using the human raw_text.

    Only a word-for-word replacement where deleting the non-Polish letters from the raw
    word gives exactly the normalized word is accepted. Anything else is left alone.
    """
    a, b = words(raw), words(norm)
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for rw, nw in zip(a[i1:i2], b[j1:j2]):
            stripped = "".join(c for c in rw if c in POLISH_LETTERS or c == "'")
            if stripped == nw and rw != nw and nw:
                pairs.append((nw, rw))
    out = norm
    for broken, fixed in pairs:
        out = re.sub(rf"(?<!\w){re.escape(broken)}(?!\w)", fixed, out, count=1)
    return out, pairs


def audio_stats(x: np.ndarray, cfg: dict) -> dict:
    frame = 400  # 25 ms
    n = len(x) // frame * frame
    fr = x[:n].reshape(-1, frame) if n else x.reshape(1, -1)
    e = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    return {
        "peak_db": float(e.max()),
        "silence_frac": float((e < e.max() - cfg["silence_frame_db"]).mean()),
        "clipped_frac": float((np.abs(x) >= cfg["clip_level"]).mean()),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "cleaning.yaml", encoding="utf-8"))
    meta = pd.read_csv(RAW / "metadata.tsv", sep="\t", keep_default_na=False, quoting=3,
                       dtype=str)
    log: list[list[str]] = []
    keep: list[dict] = []

    def drop(r, issue, detail=""):
        log.append([r["utt_id"], r["speaker_id"], issue, "removed", detail])

    for r in meta.to_dict("records"):
        r["utt_id"] = Path(r["file"]).stem
        # --- audio readable? ---
        try:
            x, sr = sf.read(RAW / r["file"], dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            drop(r, "CORRUPT_AUDIO", f"unreadable: {e}")
            continue
        if sr != SR or x.shape[1] != 1 or len(x) == 0:
            drop(r, "CORRUPT_AUDIO", f"sr={sr} ch={x.shape[1]} n={len(x)}")
            continue
        x = x[:, 0]
        dur = len(x) / SR
        # --- transcript checks ---
        text, raw = r["normalized_text"], r["raw_text"]
        if r["is_gold_transcript"] != "True":
            drop(r, "LOW_VALIDATION", "is_gold_transcript=False")
            continue
        if not text.strip():
            drop(r, "EMPTY_TRANSCRIPT")
            continue
        if not raw.strip():
            drop(r, "MISSING_ORTH", "raw_text empty")
            continue
        if re.search(r"\d", raw):
            drop(r, "NUM_UNVERIFIED", raw[:120])
            continue
        if unicodedata.normalize("NFC", text) != text:
            text = unicodedata.normalize("NFC", text)
            log.append([r["utt_id"], r["speaker_id"], "UNICODE_FIX", "fixed", "NFC"])
        text, pairs = restore_foreign(raw, text)
        if pairs:
            log.append([r["utt_id"], r["speaker_id"], "CHARSET_RESTORED", "fixed",
                        "; ".join(f"{a}->{b}" for a, b in pairs)])
        bad = sorted({c for c in text if c.isdigit() or (c.isalpha() and not
                      unicodedata.name(c, "").startswith("LATIN"))})
        if bad:
            drop(r, "BAD_CHARSET", "".join(bad))
            continue
        if words(raw) != words(text):
            ops = [(t, " ".join(words(raw)[i1:i2]), " ".join(words(text)[j1:j2]))
                   for t, i1, i2, j1, j2 in difflib.SequenceMatcher(
                       None, words(raw), words(text), autojunk=False).get_opcodes()
                   if t != "equal"]
            drop(r, "TEXT_MISMATCH", str(ops)[:200])
            continue
        # --- audio checks ---
        if not (cfg["min_duration_s"] <= dur <= cfg["max_duration_s"]):
            drop(r, "DURATION_OUTLIER", f"{dur:.2f}s")
            continue
        st = audio_stats(x, cfg)
        if st["silence_frac"] > cfg["max_silence_fraction"] or st["peak_db"] < cfg["min_peak_dbfs"]:
            drop(r, "SILENCE", f"silent_frac={st['silence_frac']:.2f} peak={st['peak_db']:.1f}dB")
            continue
        if st["clipped_frac"] > cfg["max_clipped_fraction"]:
            drop(r, "CLIPPED", f"clipped_frac={st['clipped_frac']:.4f}")
            continue
        r.update(text=text, duration=dur,
                 cps=len(normalize(text).replace(" ", "")) / dur,
                 audio_sha=hashlib.sha256(x.tobytes()).hexdigest(),
                 segment=r["audio_id"].split("pl_")[-1])
        keep.append(r)

    # --- speaking-rate outliers (robust z over all clips that passed so far) ---
    cps = np.array([k["cps"] for k in keep])
    med = float(np.median(cps))
    sigma = 1.4826 * float(np.median(np.abs(cps - med)))
    survivors = []
    for k in keep:
        z = (k["cps"] - med) / sigma
        if abs(z) > cfg["rate_robust_z"]:
            drop(k, "RATE_MISALIGN", f"cps={k['cps']:.2f} z={z:.2f} median={med:.2f}")
        else:
            survivors.append(k)

    # --- duplicates: keep the longest copy, ties broken by utt_id ---
    def dedup(items, key, issue):
        items = sorted(items, key=lambda k: (-k["duration"], k["utt_id"]))
        seen, out = {}, []
        for k in items:
            kk = key(k)
            if kk in seen:
                drop(k, issue, f"duplicate of {seen[kk]}")
            else:
                seen[kk] = k["utt_id"]
                out.append(k)
        return out

    survivors = dedup(survivors, lambda k: k["audio_sha"], "DUP_AUDIO")
    survivors = dedup(survivors, lambda k: k["segment"], "DUP_SEGMENT")
    survivors = dedup(survivors, lambda k: (k["speaker_id"], normalize(k["text"])), "DUP_UTT")
    survivors.sort(key=lambda k: k["utt_id"])

    # --- write cleaned set ---
    if CLEANED.exists():
        shutil.rmtree(CLEANED)
    (CLEANED / "audio").mkdir(parents=True)
    rows = []
    for k in survivors:
        x = sf.read(RAW / k["file"], dtype="float32")[0]
        q = np.round(x.astype(np.float64) * 32768)
        assert np.abs(q - x * 32768).max() < 1e-3, f"{k['utt_id']} is not 16-bit exact"
        pcm = np.clip(q, -32768, 32767).astype(np.int16)
        rel = f"audio/{k['utt_id']}.flac"
        sf.write(CLEANED / rel, pcm, SR, subtype="PCM_16", format="FLAC")
        rows.append([k["utt_id"], rel, k["speaker_id"], k["gender"], k["source_split"],
                     f"{k['duration']:.3f}", k["text"], " ".join(k["raw_text"].split())])
    with open(CLEANED / "metadata.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("utt_id\tfile\tspeaker\tgender\tsource_split\tduration\ttext\ttext_orth\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    log.sort(key=lambda x: (x[2], x[0]))
    with open(DATA / "cleaning_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LOG_COLUMNS)
        w.writerows(log)
    removed = sum(1 for x in log if x[3] == "removed")
    print(f"raw={len(meta)} kept={len(rows)} removed={removed} "
          f"fixed={sum(1 for x in log if x[3] == 'fixed')} hours_kept="
          f"{sum(float(r[5]) for r in rows) / 3600:.2f} rate_median={med:.2f} rate_sigma={sigma:.2f}")


if __name__ == "__main__":
    main()
