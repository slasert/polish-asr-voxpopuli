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
