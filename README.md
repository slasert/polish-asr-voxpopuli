# Polish Parliamentary Speech Recognition — Group AEK (COE025 NLP, Project 3)

Automatic speech recognition for **Polish** on **European Parliament plenary speech**
(VoxPopuli). We clean the data, split it by speaker, benchmark every course baseline
(B1–B10) under one scorer, and fine-tune `openai/whisper-small` as our contribution.

**Every number in this README is generated** by `scripts/make_tables.py` from
`results/metrics/*.json` (written by `scripts/score.py` and `scripts/bootstrap.py`) or from
the data logs. Blocks between `<!-- BEGIN:… -->` and `<!-- END:… -->` are overwritten by that
script; `python scripts/make_tables.py --check` fails if they are out of date.

## 1. Group

| Member | GitHub | Contribution file |
|---|---|---|
| see `contributions/` | each member commits from their own account | `contributions/<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` |

## 2. Task and language

- **Language: Polish (pl).** Polish is the only language we checked for which *all ten*
  course baselines have a real language-specific checkpoint (including HuBERT + CTC), so no
  baseline had to be skipped. The check was done against the live model cards and the Hub API
  (`docs/language_support.md`).
- Polish is morphologically rich (7 cases, rich verb inflection) and uses 9 diacritic letters
  (ą ć ę ł ń ó ś ź ż). One wrong case ending makes a whole word wrong for WER, which makes the
  WER/CER contrast informative.
- **Not a member's mother tongue; not used by another group:** each member confirms this in
  their `contributions/` file; coordination with the other groups is logged in `docs/decisions.md`.
- **Domain bonus:** parliamentary / political speech (procedure terms, legal and economic
  vocabulary, many named entities such as MEP and commissioner names).

## 3. Dataset

| | |
|---|---|
| Source | Hugging Face `facebook/voxpopuli`, config `pl`, revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` |
| Part used | the official `validation` and `test` parquet files (our pool); the ~18.6 GB `train` part does not fit in a git repository |
| License | **CC0-1.0** (public-domain dedication), see `DATA_LICENSE.md` |
| Audio | 16 kHz mono, 32-bit float WAV (raw); 16-bit FLAC (cleaned, lossless — see Cleaning) |
| Transcripts | human: official European Parliament verbatim reports, aligned to audio by the VoxPopuli authors (`is_gold_transcript` = alignment check passed) |
| Speaker info | `speaker_id`, gender |
| Why it differs from other groups | parliamentary domain, Polish, not Common Voice (the usual choice); dataset comparison in `docs/dataset_choice.md` |

The raw data (`data/raw/`) contains every clip of the pool **byte-for-byte as stored in the
parquet files**; `data/raw/CHECKSUMS.sha256` lists their SHA-256 and `scripts/01_download.py`
shows exactly how they were extracted (it can rebuild `data/raw/`). No download is needed.

## 4. Cleaning

`python scripts/02_clean.py` turns `data/raw/` into `data/cleaned/` and logs every removal and
fix to `data/cleaning_log.csv` (`utt_id, speaker, issue, action, detail`). Thresholds live in
`configs/cleaning.yaml`. The script is deterministic (tested). Converting the float WAVs to
16-bit FLAC is **not** counted as cleaning; it is lossless here because every raw sample is an
exact multiple of 1/32768 (the script asserts this).

What we found and did:

| Check | What it catches in this data | Action |
|---|---|---|
| `NUM_UNVERIFIED` | `raw_text` has digits. VoxPopuli spelled the numbers out *automatically*, and the result is often wrong for spoken Polish: wrong case ("w dwa tysiące jedenaście roku" for *jedenastego*), misspellings ("dziewięćdzisiąt"), or nonsense ("2008/977/WSiSW" → "dwa tysiące osiem dziewięćset siedemdziesiąt siedem wsis"). We cannot verify these references without listening, and we do not "fix" references with a model. | remove |
| `CHARSET_RESTORED` | VoxPopuli's normalizer **deleted** letters outside the Polish alphabet: "Pöttering" → "pttering", "Šefčovič" → "efovi", "Cancún" → "cancn". We restore the word from the human `raw_text`, only when deleting the foreign letters from the raw word gives exactly the broken word. | fix |
| `DUP_SEGMENT` | the same plenary segment (same date-time-index) published twice under two session ids (`…PLENARY-4-pl_…` and `…PLENARY-pl_…`), sometimes with one copy truncated | keep the longest copy |
| `DUP_UTT` | same speaker + same transcript (overlapping segments of one speech) | keep the longest copy |
| `MISSING_ORTH` | empty `raw_text`, so no orthographic reference (and in one case the normalized text still contained "o 1 1 b") | remove |
| `TEXT_MISMATCH` | `raw_text` and `normalized_text` disagree on words (e.g. "in" dropped, "czy" vs. "obamacare") — one of them does not match the audio, and we cannot tell which | remove |
| `DURATION_OUTLIER` | < 1.0 s (single interjections, often cut mid-word) or > 30 s (Whisper's 30 s window; bounds GPU memory for training) | remove |
| `RATE_MISALIGN` | characters per second more than 3.5 robust standard deviations (1.4826·MAD) from the median: the transcript covers only part of the audio, or more than the audio | remove |
| `SILENCE` | more than half of the 25 ms frames are 40 dB below the clip's loudest frame, or the loudest frame is below −40 dBFS | remove |
| `CLIPPED`, `CORRUPT_AUDIO`, `DUP_AUDIO`, `LOW_VALIDATION`, `EMPTY_TRANSCRIPT`, `BAD_CHARSET`, `UNICODE_FIX` | checked; nothing in this data triggered them (all clips are `is_gold_transcript`, NFC, readable, < 0.1 % clipped samples) | — |

Summary of `data/cleaning_log.csv`:

<!-- BEGIN:cleaning -->
| Reason code | Action | Clips | Hours |
|---|---|---|---|
| `NUM_UNVERIFIED` | removed | 227 | 0.804 |
| `DUP_SEGMENT` | removed | 77 | 0.211 |
| `DURATION_OUTLIER` | removed | 37 | 0.129 |
| `MISSING_ORTH` | removed | 34 | 0.123 |
| `DUP_UTT` | removed | 15 | 0.010 |
| `RATE_MISALIGN` | removed | 11 | 0.020 |
| `TEXT_MISMATCH` | removed | 8 | 0.018 |
| `SILENCE` | removed | 7 | 0.005 |
| `CHARSET_RESTORED` | fixed | 15 | 0.065 |
| **total removed** | | **416** of 3522 | **1.320** of 9.88 |
<!-- END:cleaning -->

**Manual audit:** not done yet — listening to a random sample of 100 cleaned clips and
recording the label error rate is assigned to the data owner (see "Open items").

## 5. Splits

`scripts/03_make_splits.py` assigns **whole speakers** to train/dev/test (80/10/10 **by hours**),
shuffling speakers with the global seed (`configs/seeds.yaml`, 1234) separately per gender so dev
and test keep the corpus gender mix; a speaker with more audio than half the dev target goes to
train. Every VoxPopuli clip is a segment of one speaker's speech in one session, so speaker groups
also keep each recording in one split (tested). `scripts/04_split_stats.py` asserts disjointness
and counts sentence overlap.

<!-- BEGIN:split_stats -->
Generated by `scripts/04_split_stats.py`. Speakers are pairwise disjoint (asserted).

| Split | Speakers | Utterances | Hours | Female hours share |
|---|---|---|---|---|
| train | 68 | 2454 | 6.78 | 15.7% |
| dev | 18 | 316 | 0.90 | 14.2% |
| test | 16 | 336 | 0.88 | 12.0% |
| **total** | 102 | 3106 | 8.56 | |

**dev sentences found word-for-word in train (after normalization):** 9 of 316 (2.85%)

**test sentences found word-for-word in train (after normalization):** 4 of 336 (1.19%)

**VoxPopuli's own split (checked, not used):** validation has 48 speakers, test has 89, and 35 speakers appear in both, so the official split is not speaker-disjoint. We re-split.
<!-- END:split_stats -->

## 6. Metrics and normalization

Following the Hugging Face Audio Course, chapter 5 "Evaluation"
(<https://huggingface.co/learn/audio-course/en/chapter5/evaluation>):

- **WER** (word error rate, (S + D + I) / N over words) is the standard ASR metric and measures
  what a reader of the transcript experiences: a word is right or wrong. It is our **primary
  metric**. Its weakness for Polish: one wrong case ending ("komisji" vs. "komisja") counts as a
  full word error.
- **CER** (character error rate) gives partial credit for nearly-right words, so it is more
  informative for a morphologically rich language with diacritics, where many errors are a
  single ending or a missing ogonek. We report it next to WER for every system.
- **Corpus-level pooling:** WER = Σ(S+D+I) / ΣN over the whole split (`src/eval/metrics.py`); we
  never average per-clip WERs (a 2-word clip would weigh as much as a 30-word clip; tested).
- **Normalized vs. orthographic** (the course's distinction): *normalized* scores compare
  normalized hypothesis and reference (our primary numbers); *orthographic* scores compare
  hypothesis and VoxPopuli `raw_text` with casing and punctuation kept, which shows how usable
  the raw output is. CTC/transducer baselines trained on lowercase text cannot produce casing or
  punctuation, so their orthographic WER is high by construction.
- **Normalization policy** (`src/normalize.py`, one function for every system): NFC → lowercase
  → digits written out as Polish cardinals with `num2words` (our references contain no digits, so
  this only stops systems from being punished for writing "2015") → letters with non-Polish
  diacritics reduced to the base letter (ą ć ę ł ń ó ś ź ż kept) → apostrophes deleted →
  hyphens, dashes and all other punctuation become spaces → whitespace collapsed. We do not use
  Whisper's `EnglishTextNormalizer`, which applies English spelling rules.
- **Secondary:** S/D/I counts, MER and WIL are stored in every metrics JSON; RTFx (audio seconds
  / compute seconds, model loading excluded) with the hardware named.
- **Paired bootstrap** (`scripts/bootstrap.py`): 2,000 resamples of test utterances, the same
  draw for both systems, re-pooled WER/CER; we report the 95 % percentile interval of the
  difference.

## 7. Baselines

All systems are scored by `scripts/score.py` on the same test clips with the same references and
normalizer. Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11.

### Table A — pretrained baselines (no training by us)

<!-- BEGIN:table_a -->
| System | Family | Exact checkpoint @ revision | Language setting | Decoding | Dev WER | Test WER (norm) | Test CER (norm) | Test WER (orth) | Test CER (orth) | RTFx |
|---|---|---|---|---|---|---|---|---|---|---|
| `wav2vec2_xlsr53_pl` | B5 SSL + CTC | [`jonatasgrosman/wav2vec2-large-xlsr-53-polish`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish/tree/6b1cea36bd8bc5f65ec8081667cd9c0207d51970) @ `6b1cea36bd` | pl (model is pl-only) | greedy CTC (argmax), no language model | 35.12 | **31.23** | 9.16 | 46.37 | 12.24 | 147.81 |
| `hubert_large_pl_s995` | B6 SSL + CTC | [`jonatasgrosman/exp_w2v2t_pl_hubert_s995`](https://huggingface.co/jonatasgrosman/exp_w2v2t_pl_hubert_s995/tree/f304ca6d57ffa98bed7197dc243d09864dd4eb37) @ `f304ca6d57` | pl (model is pl-only) | greedy CTC (argmax), no language model | 65.60 | **61.52** | 15.51 | 70.69 | 18.37 | 146.99 |
| `whisper_small` | B7 pretrained AED | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) @ `973afd2496` | language='polish', task='transcribe' | greedy (num_beams=1), no timestamps | 19.93 | **16.38** | 8.66 | 22.68 | 9.70 | 33.64 |
| `canary_1b_v2` | B8 pretrained AED | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2/tree/d455706339a6b32e1aa40f82c713a482a0c938e2) @ `d455706339` | source_lang='pl', target_lang='pl' | model-card default (beam search as configured in the .nemo, pnc=yes); timestamps model not loaded (restore_timestamps_model=False) | 8.64 | **7.08** | 3.72 | 10.50 | 4.16 | 40.26 |
| `parakeet_tdt_0.6b_v3` | B9 pretrained transducer | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b) @ `541d1f99c6` | no language argument: the model detects the language itself (see model card) | model-card default (greedy TDT) | 8.92 | **6.60** | 3.41 | 9.90 | 3.83 | 122.82 |
| `qwen3_asr_0.6b` | B10 pretrained audio-LLM ASR | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) @ `5eb144179a` | language='Polish' | model-card default (transformers backend, greedy) | 26.40 | **23.99** | 9.61 | 30.56 | 10.82 | 11.49 |
<!-- END:table_a -->

B6 has three published HuBERT + CTC checkpoints for Polish (same recipe, different seeds). We
picked the one with the lowest **dev** WER; only that one was run on test:

<!-- BEGIN:hubert_dev -->
| HuBERT checkpoint (B6 candidates) | Dev WER | Dev CER | Used on test |
|---|---|---|---|
| [`jonatasgrosman/exp_w2v2t_pl_hubert_s6`](https://huggingface.co/jonatasgrosman/exp_w2v2t_pl_hubert_s6/tree/1c177047c5f74061be360783af9c8f30d842283f) @ `1c177047c5` | 67.75 | 18.76 | no |
| [`jonatasgrosman/exp_w2v2t_pl_hubert_s484`](https://huggingface.co/jonatasgrosman/exp_w2v2t_pl_hubert_s484/tree/0c43b3d301a2b4336fa6f42e7145765b841c9127) @ `0c43b3d301` | 66.27 | 18.42 | no |
| [`jonatasgrosman/exp_w2v2t_pl_hubert_s995`](https://huggingface.co/jonatasgrosman/exp_w2v2t_pl_hubert_s995/tree/f304ca6d57ffa98bed7197dc243d09864dd4eb37) @ `f304ca6d57` | 65.60 | 18.16 | yes (lowest dev WER) |
<!-- END:hubert_dev -->

Notes: Whisper is multilingual `openai/whisper-small` (not `.en`). Parakeet is **v3**; it has no
language argument (it identifies the language itself, per its model card). Qwen3-ASR uses the
0.6B variant: the 1.7B model does not fit in 4 GB in bf16. The wav2vec 2.0 repository also ships
an n-gram LM; we deliberately decode **without** it (vanilla greedy CTC), like the HuBERT models.
Canary/Parakeet list `nvidia/Granary` as training data; European Parliament audio may have been
seen by these models (see Limitations). Canary's `.nemo` file also contains a separate 2.5 GB CTC
model that NeMo uses only for word timestamps and for chunking a single long file; with it, the
model does not fit in our 8 GB of RAM (the restore crashed), so we load Canary with
`restore_timestamps_model=False`. We use neither feature (batched clips of at most 30 s), so the
transcription is the model-card default. The `.nemo` archives are unpacked once into the Hugging
Face cache and their weights memory-mapped (`baselines/pretrained/systems.py`).

### Table B — baselines we trained ourselves (vanilla architectures, our train split only)

<!-- BEGIN:table_b -->
| System | Family | Toolkit / recipe | Features | Train hours | Selected (on dev) | Decoding | Dev WER | Test WER | Test CER | Test WER (orth) | RTFx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gmm_hmm` | B1 classical MFCC + GMM-HMM | own Python implementation (numpy/numba/torch) | 13 MFCC + d + dd (39), per-speaker CMVN | 6.78 | LM weight + insertion bonuses (grid) | Viterbi grapheme loop, char bigram LM, lm_weight=10.0, unit_insertion_bonus=2.0, word_boundary_bonus=8.0 (chosen on dev) | 98.45 | **98.14** | 50.24 | 98.38 | 538.45 |
| `e2e_ctc` | B2 CTC encoder (Conformer, from scratch) | own PyTorch recipe (torchaudio Conformer) | 80-dim log-mel, global CMVN, SpecAugment | 6.78 | epoch 60 of 60 | greedy CTC | 73.85 | **70.70** | 20.26 | 79.09 | 685.83 |
| `e2e_aed` | B3a attention encoder-decoder (Conformer + Transformer decoder, from scratch) | own PyTorch recipe (torchaudio Conformer) | 80-dim log-mel, global CMVN, SpecAugment | 6.78 | epoch 40 of 60 | attention beam search (beam 5, length-normalized) | 101.33 | **102.00** | 81.92 | 102.17 | 7.13 |
| `e2e_hybrid` | B3b hybrid CTC/attention (Conformer + Transformer decoder, from scratch) | own PyTorch recipe (torchaudio Conformer) | 80-dim log-mel, global CMVN, SpecAugment | 6.78 | epoch 60 of 60 | attention beam search (beam 5, length-normalized) + CTC rescoring of the final beam (weight 0.3) | 74.24 | **72.56** | 40.23 | 82.69 | 5.27 |
| `e2e_rnnt` | B4 transducer RNN-T (Conformer + LSTM predictor, from scratch) | own PyTorch recipe (torchaudio Conformer) | 80-dim log-mel, global CMVN, SpecAugment | 6.78 | epoch 50 of 60 | greedy RNN-T (max 5 symbols/frame) | 60.89 | **55.72** | 21.14 | 67.02 | 40.19 |
<!-- END:table_b -->

- **B1 MFCC + GMM-HMM** (`baselines/trained/gmm_hmm.py`). Kaldi needs Linux; on our machine the
  Docker engine was broken and WSL ran in safe mode without networking, so we implemented the
  classical recipe ourselves: 13 Kaldi-compatible MFCC + Δ + ΔΔ, per-speaker CMVN, 3-state
  left-to-right grapheme HMMs (Polish spelling is close to phonemic; no pronunciation lexicon was
  available), a word-boundary unit and a silence unit, diagonal GMMs with flat start, Viterbi
  training and mixture splitting up to 16 Gaussians per state, and Viterbi decoding over a
  grapheme loop with a character bigram LM trained on train transcripts only. LM weight, a
  per-grapheme insertion bonus and a word-boundary bonus (the analogue of Kaldi's word insertion
  penalty) were chosen on dev from a 100-point grid (stored in the run info JSON). Unlike Kaldi
  `tri1` it has no triphones, no lexicon and no word-level LM, so it often merges or splits words:
  its CER is meaningful, its WER is close to 100 %.
- **B2–B4 neural families** (`baselines/trained/e2e.py`, configs `configs/e2e_*.yaml`). One
  recipe for all: 80-dim log-mel, global mean/variance from train, SpecAugment, 4× convolutional
  subsampling and an 8-layer Conformer encoder (d = 144), characters as output units. B2 CTC
  (`ctc_weight` 1.0); B3a pure attention encoder-decoder (`ctc_weight` 0.0); B3b hybrid
  CTC/attention (loss 0.3·CTC + 0.7·attention, CTC rescoring of the final beam); B4 RNN-T (LSTM
  prediction network + joint network, `torchaudio` RNN-T loss). The checkpoint is the epoch
  with the lowest **dev** WER under fast greedy decoding (for the hybrid: greedy decoding of its CTC
  head, because greedy attention decoding loops early in training; `docs/decisions.md` D8). These
  models are trained from scratch on under 7 hours, so high error rates are expected; we did not
  add pretrained weights, which would make them different systems. The pure attention model (B3a)
  did not learn a usable alignment on this little data: its decoder repeats or drops text, so its
  WER is around 100 %. Adding the CTC loss (B3b hybrid) is exactly the fix for this.

### Table D — not run

None: every course baseline supports Polish (`docs/language_support.md`).

## 8. Our contribution — fine-tuning `openai/whisper-small` on our train split

`contribution/train_whisper.py`, config `configs/contrib_whisper.yaml`: full fine-tuning with
Hugging Face `Seq2SeqTrainer`, Adafactor (AdamW's optimizer state does not fit in 4 GB next to the
model), fp16 autocast, gradient checkpointing, seed 1234, target = the cased and punctuated
transcript, dev WER (our normalizer, greedy decoding) every 150 steps, early stopping, and the
checkpoint with the lowest dev WER kept. The test set was decoded once, with the selected
checkpoint and exactly the decoding of B7, so the comparison isolates our training. The ΔWER
interval is the paired bootstrap against B7 (`openai/whisper-small` without our training).

<!-- BEGIN:table_c -->
| System | Base checkpoint @ revision | What we did | Train hours | Steps | Dev WER | Test WER | Test CER | Test WER (orth) | ΔWER vs. base (95% CI) | ΔCER vs. base (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| `whisper_small_ft` | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) @ `973afd2496` | full fine-tuning, adafactor, lr 1e-05, effective batch 16, early stopping on dev WER | 6.78 | 750 | 18.84 | **17.91** | 9.25 | 23.11 | 1.53 [0.59, 2.45] | 0.59 [0.08, 1.10] |
<!-- END:table_c -->

**Result: the fine-tuning did not help on test.** The checkpoint selected on dev (step 300, about
two epochs) lowered dev WER but raised test WER, and the paired bootstrap interval of that
difference lies entirely above zero (tables above and below). The error breakdown in Section 9
("Fine-tuned vs. base Whisper") shows why: on **both** dev and test, fine-tuning added
substitutions; on dev it also removed many insertions (the base model repeats or hallucinates
text on a few dev clips), and that is where the dev gain came from. On test there were no such
insertions to remove, and the extra errors are spread over many clips rather than a few
outliers. Our reading: 6.78 hours from 68 speakers (Section 5) is too little to adapt a 244M-parameter model without it drifting toward the training
speakers' wording, and a 316-clip dev set can select a checkpoint for the wrong reason (a few
fixed hallucinations). We report the result as it is; we did not change anything after seeing
the test numbers. What we would try next (not done): freezing the encoder, a lower learning rate,
or more training data from VoxPopuli `train`.

Paired bootstrap results (A − B; negative = A is better): our model against its own starting
checkpoint (`whisper_small`), and against the baseline with the lowest dev WER (`canary_1b_v2`):

<!-- BEGIN:bootstrap -->
| A | B | Split | ΔWER (A−B) | 95% CI | ΔCER (A−B) | 95% CI | Resamples |
|---|---|---|---|---|---|---|---|
| `whisper_small_ft` | `canary_1b_v2` | test | 10.83 | [6.63, 18.72] | 5.52 | [2.36, 11.69] | 2000 |
| `whisper_small_ft` | `whisper_small` | test | 1.53 | [0.59, 2.45] | 0.59 | [0.08, 1.10] | 2000 |
<!-- END:bootstrap -->

## 9. Error analysis and limitations

`python analysis/error_analysis.py` produces everything below; the tables are pasted by
`scripts/make_tables.py`. What we read from them:

- **WER vs. CER for Polish.** For every system, CER is far below WER. For the strong systems
  about a quarter of all word substitutions are *inflection* errors (right stem, wrong case or verb
  ending), and only a few percent differ in diacritics alone. WER counts each of these as a full
  word error, which is why we report CER next to it (Section 6).
- **Domain and training data matter more than architecture.** The two SSL + CTC models were
  fine-tuned by their authors on Common Voice read speech; on parliamentary speech their CER is
  moderate but their WER is high (many nearly right words). Models trained on large, mixed data
  (Whisper, Qwen3-ASR) do much better, and the NVIDIA models, whose training data includes
  European Parliament audio, do best (see Limitations).
- **Hallucinations in attention decoders.** `whisper_small` has its worst WER on 1–5 s clips. Almost
  all of those errors are insertions, mostly from one clip where it loops ("nieco nieco nieco …",
  see `analysis/worst_whisper_small.txt`). Our from-scratch attention models show the same
  failure on long clips: the hybrid is near 100 % WER on 20–30 s clips, while the CTC and RNN-T
  models, which cannot loop, stay flat across durations. Pure AED never learned a stable alignment.
- **Gender.** Most systems are worse on female speakers, who are a minority in every split
  (Section 5, female hours share); the test set has few female speakers, so these numbers are noisy.
- **Sentence overlap is not a factor.** Only a handful of test sentences occur in train, and
  removing them changes no system's WER noticeably (last table).
- **Domain terms.** Key-term recall (parliamentary vocabulary) follows overall WER; our fine-tuning
  raised it slightly even though overall test WER got worse.

<!-- BEGIN:analysis -->
**Error types on test** (normalized text). *Inflection* = substitution where both words share a stem of at least 4 letters and differ only in the last ≤ 3 letters (a wrong case or verb ending); *diacritic only* = the words are equal once ą ć ę ł ń ó ś ź ż lose their diacritics. Key-term recall = share of reference words starting with one of the parliamentary stems in `configs/domain_terms.yaml` that the system got exactly right.

| System | Substitutions | Deletions | Insertions | Inflection share of subs | Diacritic-only share of subs | Key-term recall (n) |
|---|---|---|---|---|---|---|
| `canary_1b_v2` | 209 | 128 | 121 | 27.75 % | 1.91 % | 95.28 % (339) |
| `e2e_aed` | 4111 | 2196 | 287 | 0.83 % | 0.02 % | 0.00 % (339) |
| `e2e_ctc` | 3575 | 692 | 304 | 14.32 % | 0.62 % | 54.57 % (339) |
| `e2e_hybrid` | 2626 | 1020 | 1045 | 13.71 % | 1.26 % | 59.00 % (339) |
| `e2e_rnnt` | 2714 | 593 | 295 | 13.67 % | 0.81 % | 69.62 % (339) |
| `gmm_hmm` | 3750 | 2570 | 25 | 1.25 % | 0.03 % | 0.00 % (339) |
| `hubert_large_pl_s995` | 3032 | 566 | 379 | 16.79 % | 1.62 % | 36.28 % (339) |
| `parakeet_tdt_0.6b_v3` | 199 | 115 | 113 | 28.64 % | 4.52 % | 95.28 % (339) |
| `qwen3_asr_0.6b` | 1121 | 244 | 186 | 22.93 % | 2.41 % | 71.68 % (339) |
| `wav2vec2_xlsr53_pl` | 1366 | 349 | 304 | 23.06 % | 2.34 % | 44.54 % (339) |
| `whisper_small` | 469 | 209 | 381 | 27.51 % | 2.77 % | 86.43 % (339) |
| `whisper_small_ft` | 530 | 216 | 412 | 25.66 % | 3.21 % | 89.68 % (339) |

**Test WER (%) by clip duration** (test clips per bucket: 1-5 s: 77, 5-10 s: 133, 10-20 s: 108, 20-30 s: 18):

| System | 1-5 s | 5-10 s | 10-20 s | 20-30 s |
|---|---|---|---|---|
| `canary_1b_v2` | 6.45 | 7.25 | 6.87 | 8.08 |
| `e2e_aed` | 137.20 | 100.58 | 97.78 | 97.80 |
| `e2e_ctc` | 76.61 | 71.85 | 68.91 | 70.78 |
| `e2e_hybrid` | 57.27 | 58.84 | 78.60 | 98.68 |
| `e2e_rnnt` | 61.14 | 54.76 | 54.86 | 58.30 |
| `gmm_hmm` | 99.63 | 98.85 | 97.56 | 97.50 |
| `hubert_large_pl_s995` | 61.88 | 59.75 | 62.42 | 62.41 |
| `parakeet_tdt_0.6b_v3` | 6.26 | 5.09 | 7.06 | 9.40 |
| `qwen3_asr_0.6b` | 23.39 | 23.78 | 23.24 | 28.63 |
| `wav2vec2_xlsr53_pl` | 33.89 | 30.26 | 31.05 | 32.89 |
| `whisper_small` | 54.14 | 13.74 | 12.19 | 13.80 |
| `whisper_small_ft` | 53.78 | 13.40 | 14.78 | 17.62 |

**Test WER (%) by speaker gender:**

| System | female | male |
|---|---|---|
| `canary_1b_v2` | 10.80 | 6.60 |
| `e2e_aed` | 101.21 | 102.10 |
| `e2e_ctc` | 75.03 | 70.14 |
| `e2e_hybrid` | 84.08 | 71.07 |
| `e2e_rnnt` | 61.40 | 54.98 |
| `gmm_hmm` | 98.92 | 98.04 |
| `hubert_large_pl_s995` | 63.29 | 61.29 |
| `parakeet_tdt_0.6b_v3` | 10.66 | 6.08 |
| `qwen3_asr_0.6b` | 26.05 | 23.72 |
| `wav2vec2_xlsr53_pl` | 35.76 | 30.64 |
| `whisper_small` | 16.33 | 16.39 |
| `whisper_small_ft` | 17.81 | 17.92 |

**Cost on test** (RTFx = audio seconds / compute seconds; RTX 3050 Laptop 4 GB):

| System | RTFx | Peak GPU memory (MB) |
|---|---|---|
| `canary_1b_v2` | 40.26 | 2231.9 |
| `e2e_aed` | 7.13 | 2843.0 |
| `e2e_ctc` | 685.83 | 2836.4 |
| `e2e_hybrid` | 5.27 | 2843.0 |
| `e2e_rnnt` | 40.19 | 2838.0 |
| `gmm_hmm` | 538.45 | 68.0 |
| `hubert_large_pl_s995` | 146.99 | 2031.1 |
| `parakeet_tdt_0.6b_v3` | 122.82 | 3164.7 |
| `qwen3_asr_0.6b` | 11.49 | 2920.6 |
| `wav2vec2_xlsr53_pl` | 147.81 | 2031.1 |
| `whisper_small` | 33.64 | 1072.3 |
| `whisper_small_ft` | 55.52 | 1072.3 |

**Fine-tuned vs. base Whisper** (`whisper_small_ft` − `whisper_small`, normalized text; a clip is *worse* if it has more word errors after fine-tuning):

| Split | WER base → fine-tuned | Δ substitutions | Δ deletions | Δ insertions | Clips worse | Clips better | Clips equal |
|---|---|---|---|---|---|---|---|
| dev | 19.93 → 18.84 | +60 | +14 | -147 | 131 | 63 | 122 |
| test | 16.38 → 17.91 | +61 | +7 | +31 | 97 | 73 | 166 |

Per-system top-20 substitution pairs: `analysis/substitutions_<system>.csv`; alignments of the 10 worst clips: `analysis/worst_<system>.txt`.
<!-- END:analysis -->

Test WER on the full test set and on test sentences that do not occur word-for-word in train:

<!-- BEGIN:unseen -->
| System | Test WER (all) | Test WER (sentences not in train) | Utterances not in train |
|---|---|---|---|
| `wav2vec2_xlsr53_pl` | 31.23 | 31.18 | 332 |
| `hubert_large_pl_s995` | 61.52 | 61.45 | 332 |
| `whisper_small` | 16.38 | 16.31 | 332 |
| `canary_1b_v2` | 7.08 | 7.06 | 332 |
| `parakeet_tdt_0.6b_v3` | 6.60 | 6.58 | 332 |
| `qwen3_asr_0.6b` | 23.99 | 23.90 | 332 |
| `gmm_hmm` | 98.14 | 98.11 | 332 |
| `e2e_ctc` | 70.70 | 70.67 | 332 |
| `e2e_aed` | 102.00 | 101.75 | 332 |
| `e2e_hybrid` | 72.56 | 72.57 | 332 |
| `e2e_rnnt` | 55.72 | 55.66 | 332 |
| `whisper_small_ft` | 17.91 | 17.86 | 332 |
<!-- END:unseen -->

**Limitations.** (1) Our data is ~8.6 hours after cleaning; trained-from-scratch baselines are
data-starved. (2) Pretrained models may have seen European Parliament audio (Granary for
Canary/Parakeet; unknown for Whisper and Qwen3-ASR), so their test numbers can be optimistic. The
NVIDIA models are far ahead of every other system here, and the
[Granary dataset page](https://huggingface.co/datasets/nvidia/Granary) lists a `pl_voxpopuli`
subset (European Parliament plenary audio, checked 2026-11-20). So these two numbers most likely
measure in-domain, possibly already seen audio and should not be read as a fair comparison. Our fine-tuning comparison (same base
model, same test) is not affected by this. (3) We removed
clips with numbers in the original text, so the test set under-represents numbers. (4) VoxPopuli
transcripts come from official verbatim reports, which are lightly edited (hesitations and
false starts are not written); some "errors" are the systems transcribing what was really said.
(5) B1 is a Python re-implementation, not a Kaldi recipe (see Section 7). (6) The test set has 16
speakers, so confidence intervals are wide.

## 10. Reproduce

Environment: Python 3.11, CUDA 12.4, `pip install -r requirements.txt` (all versions pinned;
`torch`/`torchaudio` from `https://download.pytorch.org/whl/cu124`). GPU used: RTX 3050 Laptop
4 GB. Run from the repository root:

```bash
python scripts/01_download.py --cache /tmp/vp     # optional: rebuilds data/raw (already committed)
python scripts/02_clean.py                         # data/raw -> data/cleaned + data/cleaning_log.csv
python scripts/03_make_splits.py                   # data/splits/{train,dev,test}.tsv
python scripts/04_split_stats.py                   # results/tables/split_stats.md
# pretrained baselines (B5-B10): vanilla, nothing tuned, so dev and then test per model
# (only the best-on-dev HuBERT seed is run on test; docs/decisions.md D9)
for s in wav2vec2_xlsr53_pl hubert_large_pl_s6 hubert_large_pl_s484 hubert_large_pl_s995 \
         whisper_small canary_1b_v2 parakeet_tdt_0.6b_v3 qwen3_asr_0.6b; do
  python scripts/run_baseline.py --system $s --split dev && python scripts/score.py --system $s --split dev
done
# trained baselines (B1-B4)
python -m baselines.trained.gmm_hmm train
python -m baselines.trained.gmm_hmm decode --split dev     # chooses LM weight / penalty on dev
for c in ctc aed hybrid rnnt; do
  python -m baselines.trained.e2e train  --config configs/e2e_$c.yaml
  python -m baselines.trained.e2e decode --config configs/e2e_$c.yaml --split dev
done
# contribution
python contribution/train_whisper.py train  --config configs/contrib_whisper.yaml
python contribution/train_whisper.py decode --config configs/contrib_whisper.yaml --split dev
# test, once, after everything above is frozen (same commands with --split test), then:
python scripts/score.py --system <name> --split test       # for every system
python scripts/bootstrap.py --a whisper_small_ft --b whisper_small --split test
python scripts/bootstrap.py --a whisper_small_ft --b canary_1b_v2 --split test           # best baseline on dev
python scripts/collect_training_logs.py                    # exp/*/train_info.json -> results/training_logs/
python analysis/error_analysis.py
python scripts/make_tables.py
```

Model weights are not committed (they are hundreds of MB to GB); the commands above regenerate
them into `exp/` (git-ignored). Predictions for every system and split are committed in
`results/predictions/`, so all scores can be recomputed with `scripts/score.py` without a GPU.

## 11. Repository map

| Path | What it is |
|---|---|
| `README.md` | this report |
| `LICENSE` | MIT license for **our code** |
| `DATA_LICENSE.md` | license and attribution of the VoxPopuli data (CC0-1.0) |
| `requirements.txt` | pinned Python packages (one environment for all systems) |
| `pytest.ini` | test configuration (`slow` marker; no cache folder) |
| `.gitignore` | keeps caches, weights and `exp/` out of git |
| `configs/` | global seed, cleaning thresholds, training configs for B2–B4 and the contribution |
| `data/raw/` | raw clips exactly as in the parquet files, `metadata.tsv`, `CHECKSUMS.sha256` |
| `data/cleaned/` | cleaned 16-bit FLAC clips and `metadata.tsv` (normalized + orthographic transcript) |
| `data/splits/` | `train.tsv`, `dev.tsv`, `test.tsv` (utt_id, path, speaker, gender, duration, text, text_orth) |
| `data/cleaning_log.csv` | every removed or fixed clip with reason code and detail |
| `scripts/` | data pipeline (`01`–`04`), `run_baseline.py`, `score.py`, `bootstrap.py`, `collect_training_logs.py`, `make_tables.py`, `check_submission.py` |
| `src/` | `normalize.py` (the one normalizer), `eval/metrics.py` (WER/CER), `asr_io.py`, `config.py` |
| `baselines/pretrained/` | B5–B10 inference code with pinned model revisions |
| `baselines/trained/` | B1 GMM-HMM and B2–B4 neural recipes |
| `contribution/` | our Whisper fine-tuning |
| `results/predictions/` | hypotheses (`<split>.tsv`) and run info (`<split>.json`) per system |
| `results/metrics/` | scores per system (`<system>.json`) and bootstrap results |
| `results/training_logs/` | per-epoch dev scores and checkpoint choices of our trained systems (copied from `exp/`) |
| `results/tables/` | generated tables (also pasted into this README) |
| `analysis/` | error-analysis script and its outputs |
| `tests/` | pytest suite for the data pipeline, splits and evaluation |
| `docs/` | decision log, language-support check, dataset comparison |
| `contributions/` | one file per member: `<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` |
| `exp/` | **not in git** (`.gitignore`): created locally by training — cached features, model weights, training logs (`train_info.json`) |

## 12. Tests

`python -m pytest` (about 2 minutes; `-m "not slow"` skips the cleaning re-run).

- `tests/test_eval.py` — the lecture's worked examples (S=1, D=1, WER 0.3333, CER 0.2273;
  WER 3.0 / MER 0.75 / WIL 0.75 with insertions), pooled corpus WER (0.20, not the mean of clip
  WERs), identical input → 0, empty hypothesis, normalizer idempotence / NFC / Polish letters /
  hyphens, per-utterance errors summing to the corpus score, bootstrap sanity.
- `tests/test_data_clean.py` — raw checksums, every cleaned clip 16 kHz mono and non-empty, no empty
  transcripts, no duplicate audio, every removed clip logged with a known reason, duration
  thresholds, no digits in references, foreign letters restored, cleaning deterministic.
- `tests/test_splits.py` — speakers and recordings disjoint, every cleaned clip in exactly one
  split, same seed → same split (and equal to the committed files), dev/test size.

## 13. License

Our code: MIT (`LICENSE`). Data: VoxPopuli, CC0-1.0, © the VoxPopuli authors / European
Parliament recordings (`DATA_LICENSE.md`). The data license does not cover our scripts, and our
code license does not cover the data.

## Open items for the team

- Fill in `contributions/` (one file per member, committed from each member's own account) and
  the members table above.
- Confirm the mother-tongue rule and that no other group uses Polish or VoxPopuli; log it in
  `docs/decisions.md`.
- Manual audit: listen to 100 random cleaned clips, record the label error rate, add it to Section 4.
- Done: every metrics JSON was re-scored after the first commit and records commit `0c43d1d`
  (only the `commit` field changed). If the code changes again, re-score the same way
  (no model is re-run; this only re-reads `results/predictions/`):
  `for s in $(ls results/predictions); do for sp in dev test; do [ -f results/predictions/$s/$sp.tsv ] && python scripts/score.py --system $s --split $sp; done; done; python scripts/make_tables.py`
- On the machine used here, `tests/test_data_clean.py::test_cleaning_is_deterministic` could not
  rewrite `data/cleaning_log.csv` (a local security filter denied Python write access to that one
  file). Determinism was checked instead by running `02_clean.py` into a scratch folder: all 3,107
  cleaned files and the log were byte-identical. Run the test on a normal machine before submitting.
