"""B2-B4 - neural ASR families trained from scratch by us on our train split only.

One shared front end and encoder so the families differ only in their output side:
  front end : 80-dim log-mel (Kaldi fbank, 25 ms / 10 ms), global mean/var from train,
              SpecAugment (2 freq masks F<=15, 2 time masks T<=40 frames) during training
  encoder   : 2 x Conv2d stride 2 (4x subsampling) + torchaudio Conformer
  vocab     : characters of the normalized train transcripts (+ space)
  B2  ctc     : CTC head, ctc_weight = 1.0, greedy decoding
  B3a aed     : Transformer decoder, ctc_weight = 0.0, beam search
  B3b hybrid  : CTC head + Transformer decoder, loss 0.3*CTC + 0.7*attention,
                beam search with CTC rescoring of the final beam (weight 0.3)
  B4  rnnt    : LSTM prediction network + joint network, RNN-T loss, greedy decoding
Checkpoint selection: the epoch with the lowest dev WER (greedy/fast decoding; greedy CTC for the
hybrid), never test.

    python -m baselines.trained.e2e train  --config configs/e2e_ctc.yaml
    python -m baselines.trained.e2e decode --config configs/e2e_ctc.yaml --split dev
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.asr_io import load_audio, read_split, write_predictions  # noqa: E402
from src.config import ROOT, set_all_seeds  # noqa: E402
from src.eval.metrics import corpus_scores  # noqa: E402
from src.normalize import normalize  # noqa: E402

DEV = torch.device("cuda")
FEATS = ROOT / "exp" / "fbank"


# ---------------------------------------------------------------- data
def fbank(x):
    w = torch.from_numpy(x).unsqueeze(0) * 32768.0
    return torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, frame_length=25.0,
                                             frame_shift=10.0, dither=0.0)


def cached_feats(split):
    FEATS.mkdir(parents=True, exist_ok=True)
    f = FEATS / f"{split}.pt"
    if not f.exists():
        df = read_split(split)
        torch.save({r.utt_id: fbank(load_audio(r.path)).half() for r in df.itertuples()}, f)
    return torch.load(f)


class Vocab:
    def __init__(self, chars):
        # 0 = blank (CTC/RNN-T) and padding, 1 = sos/eos
        self.itos = ["<b>", "<s>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, t):
        return [self.stoi[c] for c in t if c in self.stoi]

    def decode(self, ids):
        return " ".join("".join(self.itos[i] for i in ids if i > 1).split())


def batches(lengths, max_frames, shuffle, rng):
    order = np.argsort(lengths)
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > max_frames:
            out.append(cur); cur, m = [], lengths[i]
        cur.append(i); cur_max = m
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def spec_augment(x, rng):
    x = x.clone()
    T, Fd = x.shape
    for _ in range(2):
        f = rng.integers(0, 16); f0 = rng.integers(0, Fd - f + 1); x[:, f0:f0 + f] = 0
    for _ in range(2):
        t = rng.integers(0, min(41, max(1, T // 5))); t0 = rng.integers(0, max(1, T - t + 1)); x[t0:t0 + t] = 0
    return x


# ---------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, d, layers, heads, ffn, kernel, dropout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d, 3, 2), nn.ReLU(), nn.Conv2d(d, d, 3, 2), nn.ReLU())
        self.proj = nn.Linear(d * 19, d)
        self.conformer = torchaudio.models.Conformer(d, heads, ffn, layers, kernel, dropout=dropout)

    def forward(self, x, lens):
        x = self.conv(x.unsqueeze(1))  # B, d, T', 19
        b, c, t, f = x.shape
        x = self.proj(x.transpose(1, 2).reshape(b, t, c * f))
        lens = ((lens - 1) // 2 - 1) // 2
        lens = lens.clamp(min=1, max=t)
        x, lens = self.conformer(x, lens)
        return x, lens


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=2000):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class ASR(nn.Module):
    def __init__(self, cfg, V):
        super().__init__()
        d = cfg["d_model"]
        self.kind, self.V, self.ctc_weight = cfg["kind"], V, cfg.get("ctc_weight", 0.0)
        self.enc = Encoder(d, cfg["enc_layers"], cfg["heads"], cfg["ffn"], cfg["kernel"], cfg["dropout"])
        if self.kind in ("ctc", "aed") and self.ctc_weight > 0:
            self.ctc = nn.Linear(d, V)
        if self.kind == "aed":
            self.emb = nn.Embedding(V, d)
            self.pos = PosEnc(d)
            layer = nn.TransformerDecoderLayer(d, cfg["heads"], cfg["ffn"], cfg["dropout"], batch_first=True)
            self.dec = nn.TransformerDecoder(layer, cfg["dec_layers"])
            self.out = nn.Linear(d, V)
        if self.kind == "rnnt":
            j = cfg["joint_dim"]
            self.pemb = nn.Embedding(V, cfg["pred_dim"])
            self.pred = nn.LSTM(cfg["pred_dim"], cfg["pred_dim"], 1, batch_first=True)
            self.j_enc, self.j_pred = nn.Linear(d, j), nn.Linear(cfg["pred_dim"], j)
            self.j_out = nn.Linear(j, V)

    # --- attention decoder
    def att_logits(self, mem, mem_lens, ys_in):
        T = ys_in.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ys_in.device), 1)
        mpad = torch.arange(mem.shape[1], device=mem.device)[None] >= mem_lens[:, None]
        h = self.dec(self.pos(self.emb(ys_in)), mem, tgt_mask=causal, memory_key_padding_mask=mpad,
                     tgt_key_padding_mask=(ys_in == 0) & (torch.arange(T, device=ys_in.device)[None] > 0))
        return self.out(h)

    # --- transducer
    def predictor(self, ys, state=None):
        h, state = self.pred(self.pemb(ys), state)
        return h, state

    def joint(self, e, p):
        return self.j_out(torch.tanh(self.j_enc(e) + self.j_pred(p)))

    def loss(self, x, xl, ys, yl):
        mem, ml = self.enc(x, xl)
        losses = {}
        if hasattr(self, "ctc"):
            lp = F.log_softmax(self.ctc(mem).float(), -1).transpose(0, 1)
            losses["ctc"] = F.ctc_loss(lp, ys, ml, yl, blank=0, zero_infinity=True)
        if self.kind == "aed":
            sos = torch.ones(len(ys), 1, dtype=torch.long, device=ys.device)
            ys_in = torch.cat([sos, ys], 1)
            ys_out = torch.cat([ys, torch.zeros_like(sos)], 1)
            ys_out[torch.arange(len(ys)), yl] = 1  # eos
            logits = self.att_logits(mem, ml, ys_in).float()
            mask = torch.arange(ys_out.shape[1], device=ys.device)[None] <= yl[:, None]
            losses["att"] = F.cross_entropy(logits[mask], ys_out[mask], label_smoothing=0.1)
            w = self.ctc_weight
            total = w * losses["ctc"] + (1 - w) * losses["att"] if w > 0 else losses["att"]
        elif self.kind == "ctc":
            total = losses["ctc"]
        else:  # rnnt
            blank = torch.zeros(len(ys), 1, dtype=torch.long, device=ys.device)
            p, _ = self.predictor(torch.cat([blank, ys], 1))
            logits = self.joint(mem[:, :, None, :], p[:, None, :, :]).float()
            losses["rnnt"] = torchaudio.functional.rnnt_loss(
                logits, ys.int(), ml.int(), yl.int(), blank=0, reduction="mean")
            total = losses["rnnt"]
        return total, {k: float(v) for k, v in losses.items()}


# ---------------------------------------------------------------- decoding
@torch.no_grad()
def ctc_greedy(model, mem, ml):
    ids = model.ctc(mem).argmax(-1).cpu()
    out = []
    for seq, n in zip(ids, ml.cpu()):
        s = torch.unique_consecutive(seq[:n]).tolist()
        out.append([i for i in s if i != 0])
    return out


@torch.no_grad()
def ctc_logprob(model, mem, n, ys):
    lp = F.log_softmax(model.ctc(mem[None, :n]).float(), -1).transpose(0, 1)
    if len(ys) == 0:
        return float(lp[:, 0, 0].sum())
    t = torch.tensor([ys], device=mem.device)
    return -float(F.ctc_loss(lp, t, torch.tensor([n]), torch.tensor([len(ys)]), blank=0,
                             reduction="sum", zero_infinity=True))


@torch.no_grad()
def att_beam(model, mem, n, beam, maxlen, ctc_w=0.0):
    m, ml = mem[None, :n], torch.tensor([n], device=mem.device)
    hyps = [([1], 0.0)]
    done = []
    for _ in range(maxlen):
        ys = torch.tensor([h for h, _ in hyps], device=mem.device)
        lp = F.log_softmax(model.att_logits(m.expand(len(hyps), -1, -1), ml.expand(len(hyps)), ys)[:, -1].float(), -1)
        cand = []
        for (h, s), row in zip(hyps, lp):
            top = torch.topk(row, beam)
            cand += [(h + [int(i)], s + float(v)) for v, i in zip(top.values, top.indices)]
        cand.sort(key=lambda c: -c[1])
        hyps = []
        for h, s in cand[:beam]:
            (done if h[-1] == 1 else hyps).append((h, s))
        if not hyps or len(done) >= beam:
            break
    done += hyps
    def final(h, s):
        toks = [t for t in h[1:] if t != 1]
        att = s / max(1, len(toks) + 1)  # length-normalized attention score
        if ctc_w > 0:
            return (1 - ctc_w) * att + ctc_w * ctc_logprob(model, mem, n, toks) / max(1, len(toks) + 1)
        return att
    best = max(done, key=lambda hs: final(*hs))
    return [t for t in best[0][1:] if t > 1]


@torch.no_grad()
def att_greedy(model, mem, ml, maxlen):
    ys = torch.ones(len(mem), 1, dtype=torch.long, device=mem.device)
    fin = torch.zeros(len(mem), dtype=torch.bool, device=mem.device)
    for _ in range(maxlen):
        nxt = model.att_logits(mem, ml, ys)[:, -1].argmax(-1)
        nxt[fin] = 1
        ys = torch.cat([ys, nxt[:, None]], 1)
        fin |= nxt == 1
        if fin.all():
            break
    out = []
    for row in ys[:, 1:].tolist():
        out.append(row[:row.index(1)] if 1 in row else row)
    return out


@torch.no_grad()
def rnnt_greedy(model, mem, ml, max_sym=5):
    out = []
    for e, n in zip(mem, ml.tolist()):
        y = torch.zeros(1, 1, dtype=torch.long, device=mem.device)
        p, st = model.predictor(y)
        hyp = []
        for t in range(n):
            for _ in range(max_sym):
                k = int(model.joint(e[t], p[0, -1]).argmax())
                if k == 0:
                    break
                hyp.append(k)
                p, st = model.predictor(torch.tensor([[k]], device=mem.device), st)
        out.append(hyp)
    return out


def transcribe(model, cfg, feats, ids, stats, final=False):
    model.eval()
    hyps = {}
    lens = np.array([len(feats[u]) for u in ids])
    for b in batches(lens, cfg["eval_max_frames"], False, None):
        bu = [ids[i] for i in b]
        x, xl = collate([feats[u] for u in bu], stats)
        with torch.autocast("cuda", dtype=torch.float16):
            mem, ml = model.enc(x, xl)
        mem = mem.float()
        if model.kind == "ctc" or (not final and hasattr(model, "ctc")):
            # the hybrid is selected on dev with its CTC head: greedy attention decoding loops
            # early in training, which makes the per-epoch dev WER too noisy to select on
            res = ctc_greedy(model, mem, ml)
        elif model.kind == "rnnt":
            res = rnnt_greedy(model, mem, ml)
        elif final:
            res = [att_beam(model, mem[i], int(ml[i]), cfg["beam"], int(ml[i]),
                            cfg["ctc_weight"] if cfg.get("ctc_weight", 0) > 0 else 0.0)
                   for i in range(len(bu))]
        else:
            res = att_greedy(model, mem, ml, int(ml.max()))
        hyps.update({u: r for u, r in zip(bu, res)})
    return hyps


def collate(xs, stats):
    xl = torch.tensor([len(x) for x in xs])
    x = nn.utils.rnn.pad_sequence([(t.float() - stats[0]) / stats[1] for t in xs], batch_first=True)
    return x.to(DEV), xl.to(DEV)


# ---------------------------------------------------------------- train / decode
def setup(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    exp = ROOT / "exp" / cfg["name"]
    exp.mkdir(parents=True, exist_ok=True)
    train = read_split("train")
    chars = sorted({c for t in train.text.map(normalize) for c in t})
    return cfg, exp, train, Vocab(chars)


def train(cfg_path):
    cfg, exp, df, vocab = setup(cfg_path)
    s = set_all_seeds()
    rng = np.random.default_rng(s)
    feats, dfeats = cached_feats("train"), cached_feats("dev")
    allf = torch.cat([f.float() for f in feats.values()])
    stats = (allf.mean(0), allf.std(0) + 1e-5)
    torch.save(stats, exp / "cmvn.pt")
    ids = list(df.utt_id)[: cfg.get("debug_utts")]  # debug_utts only in smoke-test configs
    ys = {u: vocab.encode(normalize(t)) for u, t in zip(df.utt_id, df.text)}
    dev = read_split("dev").head(cfg.get("debug_utts") or 10**9)
    dev_refs = [normalize(t) for t in dev.text]
    model = ASR(cfg, len(vocab.itos)).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=1e-3)
    scaler = torch.amp.GradScaler()
    lens = np.array([len(feats[u]) for u in ids])
    step, best, history, t0 = 0, (1e9, -1), [], time.time()
    warm = cfg["warmup_steps"]
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(lens, cfg["max_frames"], True, rng):
            bu = [ids[i] for i in b]
            x, xl = collate([spec_augment(feats[u].float(), rng) for u in bu], stats)
            y = nn.utils.rnn.pad_sequence([torch.tensor(ys[u]) for u in bu], batch_first=True).to(DEV)
            yl = torch.tensor([len(ys[u]) for u in bu], device=DEV)
            step += 1
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * min(step / warm, (warm / step) ** 0.5)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, parts = model.loss(x, xl, y, yl)
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "minutes": (time.time() - t0) / 60}
        if ep >= cfg["eval_from"] and ep % cfg["eval_every"] == 0 or ep == cfg["epochs"]:
            h = transcribe(model, cfg, dfeats, list(dev.utt_id), stats)
            hyp = [vocab.decode(h[u]) for u in dev.utt_id]
            sc = corpus_scores(dev_refs, hyp)
            rec.update(dev_wer=sc["wer"], dev_cer=sc["cer"])
            if sc["wer"] < best[0]:
                best = (sc["wer"], ep)
                torch.save(model.state_dict(), exp / "best.pt")
        history.append(rec)
        print(json.dumps(rec), flush=True)
    info = {"best_dev_wer_fast_decoding": best[0], "selected_epoch": best[1], "params": nparams,
            "train_minutes": (time.time() - t0) / 60, "history": history, "vocab": vocab.itos}
    (exp / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def decode(cfg_path, split):
    cfg, exp, df, vocab = setup(cfg_path)
    s = set_all_seeds()
    info = json.loads((exp / "train_info.json").read_text(encoding="utf-8"))
    model = ASR(cfg, len(vocab.itos)).to(DEV)
    model.load_state_dict(torch.load(exp / "best.pt"))
    stats = torch.load(exp / "cmvn.pt")
    data = read_split(split)
    feats = cached_feats(split)
    t0 = time.time()
    h = transcribe(model, cfg, feats, list(data.utt_id), stats, final=True)
    torch.cuda.synchronize()
    compute = time.time() - t0
    hyps = [vocab.decode(h[u]) for u in data.utt_id]
    if cfg["kind"] == "aed":
        dec = (f"attention beam search (beam {cfg['beam']}, length-normalized)"
               + (f" + CTC rescoring of the final beam (weight {cfg['ctc_weight']})" if cfg["ctc_weight"] > 0 else ""))
    else:
        dec = {"ctc": "greedy CTC", "rnnt": "greedy RNN-T (max 5 symbols/frame)"}[cfg["kind"]]
    run = dict(model_id=f"ours: {cfg['name']}", revision="n/a (trained from scratch)",
               family=cfg["family"], trained_by_us=True, seed=s, decoding=dec,
               toolkit="own PyTorch recipe (torchaudio Conformer)", features="80-dim log-mel, global CMVN, SpecAugment",
               train_hours=round(df.duration.sum() / 3600, 2), epochs=cfg["epochs"],
               selected_epoch=info["selected_epoch"], params=info["params"],
               train_minutes=round(info["train_minutes"], 1), config=cfg, compute_seconds=round(compute, 2))
    write_predictions(cfg["name"], split, list(data.utt_id), hyps, run)
    print("done", cfg["name"], split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "decode"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    train(a.config) if a.stage == "train" else decode(a.config, a.split)
