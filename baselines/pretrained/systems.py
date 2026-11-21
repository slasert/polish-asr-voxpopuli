"""Pretrained baselines B5-B10: inference only, vanilla settings from each model card.

Every model is loaded at a pinned Hub revision. Language is set explicitly where the
model has a language argument. Decoding is greedy unless the card says otherwise.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from src.asr_io import load_audio, timer

SYSTEMS = {
    "wav2vec2_xlsr53_pl": dict(
        model_id="jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        revision="6b1cea36bd8bc5f65ec8081667cd9c0207d51970",
        family="B5 SSL + CTC (wav2vec 2.0 XLSR-53, fine-tuned on Common Voice pl by its author)"),
    "hubert_large_pl_s6": dict(
        model_id="jonatasgrosman/exp_w2v2t_pl_hubert_s6",
        revision="1c177047c5f74061be360783af9c8f30d842283f",
        family="B6 SSL + CTC (HuBERT-large, fine-tuned on Common Voice 7 pl by its author)"),
    "hubert_large_pl_s484": dict(
        model_id="jonatasgrosman/exp_w2v2t_pl_hubert_s484",
        revision="0c43b3d301a2b4336fa6f42e7145765b841c9127",
        family="B6 SSL + CTC (HuBERT-large, fine-tuned on Common Voice 7 pl by its author)"),
    "hubert_large_pl_s995": dict(
        model_id="jonatasgrosman/exp_w2v2t_pl_hubert_s995",
        revision="f304ca6d57ffa98bed7197dc243d09864dd4eb37",
        family="B6 SSL + CTC (HuBERT-large, fine-tuned on Common Voice 7 pl by its author)"),
    "whisper_small": dict(
        model_id="openai/whisper-small",
        revision="973afd24965f72e36ca33b3055d56a652f456b4d",
        family="B7 pretrained AED (Whisper, multilingual)"),
    "canary_1b_v2": dict(
        model_id="nvidia/canary-1b-v2",
        revision="d455706339a6b32e1aa40f82c713a482a0c938e2",
        family="B8 pretrained AED (Canary, FastConformer encoder + Transformer decoder)"),
    "parakeet_tdt_0.6b_v3": dict(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        family="B9 pretrained transducer (Parakeet TDT, FastConformer)"),
    "qwen3_asr_0.6b": dict(
        model_id="Qwen/Qwen3-ASR-0.6B",
        revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        family="B10 pretrained audio-LLM ASR (Qwen3-ASR)"),
}


def _by_length(df):
    return df.sort_values("duration", ascending=False)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def run_ctc(name, df, batch_size=4):
    from transformers import AutoModelForCTC, Wav2Vec2Processor
    spec = SYSTEMS[name]
    # plain processor on purpose: the wav2vec2 repo also ships an n-gram LM, which we do not use
    proc = Wav2Vec2Processor.from_pretrained(spec["model_id"], revision=spec["revision"])
    model = AutoModelForCTC.from_pretrained(spec["model_id"], revision=spec["revision"]).cuda().eval()
    model.half()
    df = _by_length(df)
    hyps = []
    with timer() as t, torch.inference_mode():
        for i in range(0, len(df), batch_size):
            audio = [load_audio(p) for p in df.path.iloc[i:i + batch_size]]
            inp = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            mask = inp.get("attention_mask")
            logits = model(inp.input_values.cuda().half(),
                           attention_mask=None if mask is None else mask.cuda()).logits
            hyps += proc.batch_decode(logits.argmax(-1).cpu().numpy())
    run = dict(spec, decoding="greedy CTC (argmax), no language model", language="pl (model is pl-only)",
               precision="fp16", batch_size=batch_size, compute_seconds=round(t["s"], 2))
    del model
    _free()
    return list(df.utt_id), hyps, run


def run_whisper(name, df, batch_size=8):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    spec = SYSTEMS[name]
    proc = WhisperProcessor.from_pretrained(spec["model_id"], revision=spec["revision"])
    model = WhisperForConditionalGeneration.from_pretrained(
        spec["model_id"], revision=spec["revision"], torch_dtype=torch.float16).cuda().eval()
    df = _by_length(df)
    hyps = []
    with timer() as t, torch.inference_mode():
        for i in range(0, len(df), batch_size):
            audio = [load_audio(p) for p in df.path.iloc[i:i + batch_size]]
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
            out = model.generate(feats.cuda().half(), language="polish", task="transcribe",
                                 num_beams=1, do_sample=False, max_new_tokens=440)
            hyps += proc.batch_decode(out, skip_special_tokens=True)
    run = dict(spec, decoding="greedy (num_beams=1), no timestamps", language="language='polish', task='transcribe'",
               precision="fp16", batch_size=batch_size, compute_seconds=round(t["s"], 2))
    del model
    _free()
    return list(df.utt_id), hyps, run


def _nemo_model(spec, cfg_overrides=None):
    """Restore a pinned .nemo checkpoint (optionally with config keys overridden).

    The .nemo archive is unpacked once into <HF cache>/nemo-extracted/<repo>@<revision>/ and the
    weights are memory-mapped: NeMo's default restore unpacks to a temp folder and reads the
    whole state dict into RAM next to the fp32 model, which crashes (segfault) for Canary-1B on
    an 8 GB machine. The folder is outside the repository on purpose: sentencepiece cannot open
    paths with non-ASCII characters on Windows (our checkout path has some).
    """
    import tarfile
    from pathlib import Path

    from huggingface_hub.constants import HF_HUB_CACHE
    from nemo.collections.asr.models import ASRModel
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    class MmapConnector(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location="cpu"):
            return torch.load(model_weights, map_location=map_location, mmap=True)

    out = Path(HF_HUB_CACHE) / "nemo-extracted" / f"{spec['model_id'].replace('/', '--')}@{spec['revision'][:10]}"
    if not (out / "model_config.yaml").exists():
        from huggingface_hub import hf_hub_download
        fname = spec["model_id"].split("/")[1] + ".nemo"
        path = hf_hub_download(spec["model_id"], fname, revision=spec["revision"])
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tar:
            tar.extractall(out, filter="data")
    conn = MmapConnector()
    conn.model_extracted_dir = str(out)
    override = None
    if cfg_overrides:
        from omegaconf import OmegaConf
        override = OmegaConf.load(out / "model_config.yaml")
        override.update(cfg_overrides)
    return ASRModel.restore_from(str(out), map_location="cpu", save_restore_connector=conn,
                                 override_config_path=override)


def _nemo_text(o):
    return o.text if hasattr(o, "text") else str(o)


def run_canary(name, df, batch_size=4):
    spec = SYSTEMS[name]
    # The .nemo also holds a 2.5 GB CTC model used only for word timestamps and for chunking
    # single long files; we use neither (batch_size > 1, clips <= 30 s), and with it the model
    # does not fit in 8 GB of RAM, so it is not restored. Transcription is unchanged.
    model = _nemo_model(spec, {"restore_timestamps_model": False}).to(torch.bfloat16).cuda().eval()
    df = _by_length(df)
    audio = [load_audio(p) for p in df.path]
    with timer() as t, torch.inference_mode():
        out = model.transcribe(audio, batch_size=batch_size, source_lang="pl", target_lang="pl",
                               verbose=False)
    run = dict(spec, decoding="model-card default (beam search as configured in the .nemo, pnc=yes); "
                              "timestamps model not loaded (restore_timestamps_model=False)",
               language="source_lang='pl', target_lang='pl'", precision="bf16 weights",
               batch_size=batch_size, compute_seconds=round(t["s"], 2))
    hyps = [_nemo_text(o) for o in out]
    del model
    _free()
    return list(df.utt_id), hyps, run


def run_parakeet(name, df, batch_size=4):
    spec = SYSTEMS[name]
    model = _nemo_model(spec).cuda().eval()
    df = _by_length(df)
    audio = [load_audio(p) for p in df.path]
    with timer() as t, torch.inference_mode():
        out = model.transcribe(audio, batch_size=batch_size, verbose=False)
    run = dict(spec, decoding="model-card default (greedy TDT)",
               language="no language argument: the model detects the language itself (see model card)",
               precision="fp32", batch_size=batch_size, compute_seconds=round(t["s"], 2))
    hyps = [_nemo_text(o) for o in out]
    del model
    _free()
    return list(df.utt_id), hyps, run


def run_qwen(name, df, batch_size=8):
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    spec = SYSTEMS[name]
    # qwen_asr loads its processor without passing the revision on, so we load everything from
    # the local snapshot of the pinned revision instead of the repo id
    local = snapshot_download(spec["model_id"], revision=spec["revision"])
    model = Qwen3ASRModel.from_pretrained(local,
                                          dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=batch_size, max_new_tokens=256)
    df = _by_length(df)
    audio = [(load_audio(p), 16000) for p in df.path]
    with timer() as t:
        out = model.transcribe(audio=audio, language="Polish")
    run = dict(spec, decoding="model-card default (transformers backend, greedy)",
               language="language='Polish'", precision="bf16", batch_size=batch_size,
               compute_seconds=round(t["s"], 2))
    hyps = [o.text for o in out]
    del model
    _free()
    return list(df.utt_id), hyps, run


RUNNERS = {
    "wav2vec2_xlsr53_pl": run_ctc,
    "hubert_large_pl_s6": run_ctc,
    "hubert_large_pl_s484": run_ctc,
    "hubert_large_pl_s995": run_ctc,
    "whisper_small": run_whisper,
    "canary_1b_v2": run_canary,
    "parakeet_tdt_0.6b_v3": run_parakeet,
    "qwen3_asr_0.6b": run_qwen,
}
