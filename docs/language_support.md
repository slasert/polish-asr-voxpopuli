# Language support of the course baselines (checked 2026-11-20)

Checked against the live model cards / Hugging Face Hub API on 2026-11-20, not from memory.
"Yes" means the language is listed on the model card (or a checkpoint fine-tuned for that
language exists, for B5/B6).

| Baseline | Model | Polish (pl) | Hungarian (hu) | Greek (el) | Czech (cs) | Romanian (ro) |
|---|---|---|---|---|---|---|
| B5 wav2vec 2.0 + CTC | `jonatasgrosman/wav2vec2-large-xlsr-53-<lang>` | yes ([card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish)) | yes | yes | no jonatasgrosman checkpoint | no jonatasgrosman checkpoint |
| B6 HuBERT + CTC | `jonatasgrosman/exp_w2v2t_<lang>_hubert_s*` | **yes** ([card](https://huggingface.co/jonatasgrosman/exp_w2v2t_pl_hubert_s6)) | none found | only a 33-download community model | none found | none found |
| B7 Whisper | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | yes | yes | yes | yes | yes |
| B8 Canary | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | yes | yes | yes | yes | yes |
| B9 Parakeet | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | yes | yes | yes | yes | yes |
| B10 Qwen3-ASR | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | yes | yes | yes | yes | yes |

HuBERT search queries used on the Hub API: `hubert-<language>`, `exp_w2v2t_<code>_hubert`,
and all models with `pipeline_tag=automatic-speech-recognition` matching `hubert`
(top 100 by downloads). Only Polish, Dutch, Italian, Portuguese and Swedish (among the
languages that Canary, Parakeet and Qwen3-ASR all support) have a HuBERT + CTC checkpoint.

**Licenses:** Canary and Parakeet: CC-BY-4.0. Qwen3-ASR: Apache-2.0. Whisper: MIT.
jonatasgrosman checkpoints: Apache-2.0. None restricts research evaluation.

**Result:** Polish is the only candidate for which **all ten** course baselines can be run
with a real, language-specific checkpoint (no "not supported" rows needed).
