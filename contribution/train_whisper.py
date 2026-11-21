"""Our contribution: fine-tune openai/whisper-small on our Polish train split.

Model selection uses dev WER only (our normalizer), with early stopping. The test split is
decoded once, with the selected checkpoint and the same decoding as baseline B7
(greedy, language='polish', task='transcribe'), so the two are directly comparable.

    python contribution/train_whisper.py train  --config configs/contrib_whisper.yaml
    python contribution/train_whisper.py decode --config configs/contrib_whisper.yaml --split dev
    python contribution/train_whisper.py decode --config configs/contrib_whisper.yaml --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.asr_io import load_audio, read_split, timer, write_predictions  # noqa: E402
from src.config import ROOT, set_all_seeds  # noqa: E402
from src.eval.metrics import corpus_scores  # noqa: E402
from src.normalize import normalize  # noqa: E402


class SpeechSet(torch.utils.data.Dataset):
    def __init__(self, df, proc, target):
        self.df, self.proc, self.target = df.reset_index(drop=True), proc, target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        feats = self.proc.feature_extractor(load_audio(r.path), sampling_rate=16000).input_features[0]
        labels = self.proc.tokenizer(r[self.target]).input_ids
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, proc, decoder_start):
        self.proc, self.start = proc, decoder_start

    def __call__(self, items):
        batch = self.proc.feature_extractor.pad([{"input_features": x["input_features"]} for x in items],
                                                return_tensors="pt")
        lab = self.proc.tokenizer.pad([{"input_ids": x["labels"]} for x in items], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.start).all():  # the model adds the start token itself
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_cfg(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return cfg, ROOT / "exp" / cfg["name"]


def train(cfg_path):
    cfg, exp = load_cfg(cfg_path)
    s = set_all_seeds()
    proc = WhisperProcessor.from_pretrained(cfg["base_model"], revision=cfg["base_revision"],
                                            language=cfg["language"], task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"], revision=cfg["base_revision"])
    model.generation_config.language = cfg["language"]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    tr, dv = read_split("train"), read_split("dev")
    dev_refs = [normalize(t) for t in dv.text]

    def metrics(pred):
        ids = pred.predictions
        ids[ids == -100] = proc.tokenizer.pad_token_id
        hyps = proc.batch_decode(ids, skip_special_tokens=True)
        sc = corpus_scores(dev_refs, [normalize(h) for h in hyps])
        return {"wer": sc["wer"], "cer": sc["cer"]}

    args = Seq2SeqTrainingArguments(
        output_dir=str(exp), seed=s, data_seed=s,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"], warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"], optim=cfg["optim"],
        fp16=cfg["fp16"], gradient_checkpointing=cfg["gradient_checkpointing"],
        # the default reentrant checkpointing fails on Whisper ("backward through the graph a second time")
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps",
        save_steps=cfg["eval_steps"], save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False, predict_with_generate=True,
        generation_max_length=cfg["generation_max_length"], logging_steps=25,
        report_to=[], dataloader_num_workers=0, remove_unused_columns=False,
        save_only_model=True)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=SpeechSet(tr, proc, cfg["target"]),
        eval_dataset=SpeechSet(dv, proc, cfg["target"]),
        data_collator=Collator(proc, model.config.decoder_start_token_id),
        compute_metrics=metrics, processing_class=proc.feature_extractor,
        callbacks=[EarlyStoppingCallback(cfg["early_stopping_patience"])])
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(exp / "best"))
    proc.save_pretrained(str(exp / "best"))
    hist = trainer.state.log_history
    info = {"best_checkpoint": Path(trainer.state.best_model_checkpoint).relative_to(ROOT).as_posix(),
            "best_dev_wer": trainer.state.best_metric, "train_minutes": (time.time() - t0) / 60,
            "steps": trainer.state.global_step, "log_history": hist}
    (exp / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=2))


def decode(cfg_path, split, batch_size=8):
    cfg, exp = load_cfg(cfg_path)
    s = set_all_seeds()
    info = json.loads((exp / "train_info.json").read_text(encoding="utf-8"))
    proc = WhisperProcessor.from_pretrained(str(exp / "best"))
    model = WhisperForConditionalGeneration.from_pretrained(str(exp / "best"), torch_dtype=torch.float16).cuda().eval()
    df = read_split(split).sort_values("duration", ascending=False)
    hyps = []
    with timer() as t, torch.inference_mode():
        for i in range(0, len(df), batch_size):
            audio = [load_audio(p) for p in df.path.iloc[i:i + batch_size]]
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
            out = model.generate(feats.cuda().half(), language="polish", task="transcribe",
                                 num_beams=1, do_sample=False, max_new_tokens=440)
            hyps += proc.batch_decode(out, skip_special_tokens=True)
    tr = read_split("train")
    run = dict(model_id=f"ours: {cfg['base_model']} fine-tuned on our train split",
               revision=f"base {cfg['base_model']}@{cfg['base_revision']}",
               family="Contribution: fine-tuned pretrained AED (Whisper)", trained_by_us=True, seed=s,
               decoding="greedy (num_beams=1), same as B7", language="language='polish', task='transcribe'",
               precision="fp16 inference", train_hours=round(tr.duration.sum() / 3600, 2),
               best_dev_wer_during_training=info["best_dev_wer"], selected_checkpoint=info["best_checkpoint"],
               steps=info["steps"], train_minutes=round(info["train_minutes"], 1), config=cfg,
               compute_seconds=round(t["s"], 2))
    write_predictions(cfg["name"], split, list(df.utt_id), hyps, run)
    print("done", split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "decode"])
    ap.add_argument("--config", default="configs/contrib_whisper.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    train(a.config) if a.stage == "train" else decode(a.config, a.split)
