# Decisions log (Group AEK, Project 3 ASR)

Each decision has a date and a reason. Newest decisions at the bottom.

## D1 — Language: Polish (pl) — 2026-11-20
- **Why:** Polish is the only candidate we checked for which all ten course baselines (B5–B10 and
  our own B1–B4) can be run with a real language-specific checkpoint, including the hard
  HuBERT + CTC one (`jonatasgrosman/exp_w2v2t_pl_hubert_*`). See `docs/language_support.md`.
- Polish is morphologically rich (7 cases, rich verbal inflection) and has diacritics
  (ą ć ę ł ń ó ś ź ż), which makes the WER vs. CER contrast meaningful.
- **Mother-tongue rule:** to be confirmed by every member in their `contributions/` file.
  The team is at a Turkish university; Turkish was excluded up front.
- **Other groups:** to be confirmed with the other groups (record date and who confirmed here).

## D2 — Dataset: VoxPopuli, Polish transcribed subset — 2026-11-20
- Source: `facebook/voxpopuli`, config `pl`, revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`.
- **Domain:** European Parliament plenary speeches (political / parliamentary domain bonus).
- **License:** CC0-1.0 (public-domain dedication), so redistribution in a public GitHub repo is allowed.
- **Speaker IDs:** yes (`speaker_id`), plus gender. Transcripts come from the official
  European Parliament verbatim reports (human-written), aligned to audio by the VoxPopuli authors;
  `is_gold_transcript` marks segments whose alignment passed their check.
- **Pool:** we take the official `validation` and `test` parquet files (9.88 h, 3,522 clips together,
  not checked against VoxPopuli train, which we do not use) as our pool and re-split it by speaker ourselves
  (see D4). We do not use VoxPopuli `train` (~18.6 GB), because it does not fit in the repo.
- **Known risk (contamination):** the Canary-1b-v2 and Parakeet-tdt-0.6b-v3 model cards list
  `nvidia/Granary` (large pseudo-labelled multilingual corpus) as training data, and European
  Parliament audio is widely used in such corpora. We cannot exclude that some of our test audio
  was seen by these (or other) pretrained models. We report this as a limitation and do not
  "correct" for it.

## D3 — Storage: plain git, no LFS — 2026-11-20
- Audio is stored as individual small files (each well under 100 MB). Raw audio = the exact
  bytes from the parquet `audio.bytes` field, written unchanged.

## D4 — Splits: our own speaker-disjoint 80/10/10 split by hours — 2026-11-20
- VoxPopuli's own validation and test share 35 speakers (checked by `scripts/04_split_stats.py`),
  so we re-split by speaker, stratified by gender, seed 1234 (`configs/seeds.yaml`).

## D5 — Reference transcript and normalization — 2026-11-20
- Normalized reference = VoxPopuli `normalized_text` (lowercase, numbers spelled out) after our
  cleaning fixes; orthographic reference = VoxPopuli `raw_text` (cased, punctuated).
- Clips whose `raw_text` contains digits are removed: their spelled-out form was produced
  automatically and is often wrongly inflected ("w dwa tysiące jedenaście roku") or misspelled
  ("dziewięćdzisiąt"). We do not invent or model-correct references.
- One normalizer (`src/normalize.py`) for every system.

## D6 — B1 GMM-HMM implemented in Python, not Kaldi — 2026-11-20
- Kaldi needs Linux. On the training machine Docker Desktop's engine returned HTTP 500 and WSL runs
  in safe mode without networking, so neither the Kaldi image nor a Kaldi build was available.
  We implemented the classical recipe (MFCC+Δ+ΔΔ, CMVN, 3-state grapheme HMMs, diagonal GMMs,
  Viterbi training with mixture splitting, bigram LM, Viterbi decoding) in Python instead.
  Difference to a Kaldi tri1 system: no context-dependent triphones and a grapheme-level (not
  word-level) LM. Documented in the README.

## D7 — Contribution: full fine-tuning of openai/whisper-small — 2026-11-20
- Largest multilingual Whisper that fits our 4 GB GPU for full fine-tuning (with Adafactor,
  fp16 autocast and gradient checkpointing). Compared against B7 (same checkpoint, no training)
  with identical decoding, plus a paired bootstrap CI.

## D8 — Checkpoint selection for the neural baselines — 2026-11-20
- B2 CTC, B3a AED and B4 RNN-T: the epoch with the lowest dev WER under fast greedy decoding.
- B3b hybrid: the epoch with the lowest dev WER of its **CTC head** (greedy). Greedy attention decoding
  loops early in training (dev WER > 100 %), which made the per-epoch dev score useless for selection.
  The final hybrid decode is unchanged: attention beam search + CTC rescoring. Only dev is used.
- B3a pure AED did not converge on our 6.78 h of training audio (best dev WER ≈ 100 % at epoch 40 of 60;
  the decoder repeats or drops text). We report it as it is and do not tune it after the fact; the
  hybrid, which adds the CTC alignment loss, is the fix the lecture describes.

## D9 — Order of the pretrained runs — 2026-11-20
- The pretrained baselines are vanilla: nothing about them is tuned on dev, except choosing one of the
  three HuBERT seeds (lowest dev WER). So each model was run on dev and then on test right away, and its
  download was deleted afterwards (the machine had too little free disk for all checkpoints at once).
  Our own systems (B1–B4, the contribution) were decoded on test only after their training and
  dev-based selection were finished.
