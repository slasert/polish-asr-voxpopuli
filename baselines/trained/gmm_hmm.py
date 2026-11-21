"""B1 - classical MFCC + GMM-HMM recognizer, trained by us on our train split only.

Why not Kaldi: Kaldi needs Linux (Docker or WSL). Both were unavailable on our training
machine (see README, "B1"). This file implements the same classical recipe in
Python (numpy + numba for Viterbi, torch only for batched Gaussian likelihoods):

  features : 13 MFCC (Kaldi-compatible, torchaudio.compliance.kaldi.mfcc, 25 ms / 10 ms,
             23 mel bins, no energy) + delta + delta-delta = 39 dims, per-speaker CMVN
  units    : Polish graphemes (no pronunciation lexicon exists for us; Polish spelling is
             close to phonemic), a 1-state word-boundary unit "_" and a 3-state silence "#"
  HMM      : 3-state left-to-right per grapheme, self-loop + forward transitions
             estimated from the alignments
  GMM      : diagonal covariances, flat start (uniform segmentation), Viterbi training,
             mixtures split 1 -> 2 -> 4 -> 8 -> 16 per state
  LM       : grapheme bigram (add-k smoothed) on train transcripts only
  decoding : Viterbi over a grapheme loop; LM weight and insertion penalty chosen on dev

    python -m baselines.trained.gmm_hmm train
    python -m baselines.trained.gmm_hmm decode --split dev     # tunes on dev, saves choice
    python -m baselines.trained.gmm_hmm decode --split test    # uses the dev choice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.asr_io import load_audio, read_split, write_predictions  # noqa: E402
from src.config import ROOT, set_all_seeds  # noqa: E402
from src.eval.metrics import corpus_scores  # noqa: E402
from src.normalize import normalize  # noqa: E402

EXP = ROOT / "exp" / "gmm_hmm"
SIL, SP = "#", "_"
N_STATES = {SP: 1}  # everything else: 3
MIX_SCHEDULE = [1, 1, 1, 2, 2, 4, 4, 8, 8, 16, 16]  # gaussians per state at each Viterbi pass
EM_ITERS = 4
MAX_FRAMES_PER_STATE = 40000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- features
def mfcc(x: np.ndarray) -> np.ndarray:
    w = torch.from_numpy(x).unsqueeze(0) * 32768.0
    m = torchaudio.compliance.kaldi.mfcc(w, num_ceps=13, num_mel_bins=23, use_energy=False,
                                         frame_length=25.0, frame_shift=10.0, dither=0.0)
    d1 = torchaudio.functional.compute_deltas(m.T.unsqueeze(0)).squeeze(0).T
    d2 = torchaudio.functional.compute_deltas(d1.T.unsqueeze(0)).squeeze(0).T
    return torch.cat([m, d1, d2], 1).numpy().astype(np.float32)


def features(df):
    feats = {r.utt_id: mfcc(load_audio(r.path)) for r in df.itertuples()}
    for spk, g in df.groupby("speaker"):  # per-speaker CMVN
        allf = np.concatenate([feats[u] for u in g.utt_id])
        mu, sd = allf.mean(0), allf.std(0) + 1e-5
        for u in g.utt_id:
            feats[u] = (feats[u] - mu) / sd
    return feats


# ---------------------------------------------------------------- units / topology
def units_from(texts):
    letters = sorted({c for t in texts for c in t if c != " "})
    units = [SIL, SP] + letters
    n = [N_STATES.get(u, 3) for u in units]
    start = np.cumsum([0] + n[:-1]).astype(np.int64)
    return units, np.array(n, dtype=np.int64), start


def symbols(text):
    return [SIL] + [SP if c == " " else c for c in text] + [SIL]


# ---------------------------------------------------------------- GMM
class GMM:
    """Diagonal GMMs for all states: means/vars [S, M, D], log weights [S, M]."""

    def __init__(self, means, var, logw):
        self.means, self.var, self.logw = means, var, logw

    def loglik(self, x: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(x).to(DEV)
        mu = torch.from_numpy(self.means).to(DEV)
        iv = 1.0 / torch.from_numpy(self.var).to(DEV)
        const = (-0.5 * (torch.log(2 * torch.pi / iv)).sum(-1)
                 + torch.from_numpy(self.logw).to(DEV))  # [S, M]
        out = []
        for i in range(0, len(X), 4096):
            xb = X[i:i + 4096]  # [B, D]
            # -0.5 * sum (x-mu)^2 * iv  expanded
            q = ((xb ** 2) @ iv.reshape(-1, iv.shape[-1]).T
                 - 2 * xb @ (mu * iv).reshape(-1, mu.shape[-1]).T
                 + (mu ** 2 * iv).sum(-1).reshape(1, -1))
            ll = const.reshape(1, -1) - 0.5 * q  # [B, S*M]
            out.append(torch.logsumexp(ll.reshape(len(xb), *const.shape), -1).cpu())
        return torch.cat(out).numpy().astype(np.float64)


def fit_state(x: np.ndarray, m: int, init, rng) -> tuple:
    """EM for one state's diagonal GMM with m components, starting from init (or split)."""
    D = x.shape[1]
    floor = 1e-3
    if init is None or init[0].shape[0] < m:
        if init is None:
            mu, var, w = x.mean(0, keepdims=True), x.var(0, keepdims=True) + floor, np.ones(1)
        else:
            mu, var, w = init
        while mu.shape[0] < m:  # split the heaviest component
            k = int(np.argmax(w))
            eps = 0.2 * np.sqrt(var[k])
            mu = np.vstack([mu, mu[k] + eps]); mu[k] = mu[k] - eps
            var = np.vstack([var, var[k]]); w = np.append(w, w[k] / 2); w[k] /= 2
    else:
        mu, var, w = init
    X = torch.from_numpy(x).double()
    mu, var, w = torch.from_numpy(mu).double(), torch.from_numpy(var).double(), torch.from_numpy(w).double()
    for _ in range(EM_ITERS):
        ll = (-0.5 * (((X[:, None, :] - mu[None]) ** 2) / var[None] + torch.log(2 * torch.pi * var[None])).sum(-1)
              + torch.log(w[None] + 1e-12))
        g = torch.softmax(ll, 1)  # [N, M]
        nk = g.sum(0) + 1e-6
        w = nk / nk.sum()
        mu = (g.T @ X) / nk[:, None]
        var = (g.T @ X ** 2) / nk[:, None] - mu ** 2
        var = torch.clamp(var, min=floor)
    return mu.numpy(), var.numpy(), w.numpy()


# ---------------------------------------------------------------- Viterbi (numba)
@numba.njit(cache=True)
def align(ll, seq, self_lp, next_lp):
    """Forced alignment of frames to a left-to-right state sequence. Returns state per frame."""
    T, N = ll.shape[0], len(seq)
    NEG = -1e30
    d = np.full((T, N), NEG)
    bp = np.zeros((T, N), dtype=np.int8)
    d[0, 0] = ll[0, seq[0]]
    for t in range(1, T):
        for j in range(min(N, t + 1)):
            stay = d[t - 1, j] + self_lp[seq[j]]
            move = d[t - 1, j - 1] + next_lp[seq[j - 1]] if j > 0 else NEG
            if move > stay:
                d[t, j] = move + ll[t, seq[j]]
                bp[t, j] = 1
            else:
                d[t, j] = stay + ll[t, seq[j]]
    path = np.zeros(T, dtype=np.int64)
    j = N - 1
    if d[T - 1, j] <= NEG / 2:
        return path, False
    for t in range(T - 1, -1, -1):
        path[t] = seq[j]
        if bp[t, j] == 1:
            j -= 1
    return path, True


@numba.njit(cache=True)
def decode_loop(ll, ustart, ulen, self_lp, next_lp, lm, lm_start, lm_end, lmw, bonus):
    """Viterbi over a grapheme loop with a bigram LM. bonus[v] is added whenever unit v is
    entered (insertion penalty per unit; extra word-insertion term on the word boundary).
    Returns the unit sequence."""
    T, S = ll.shape
    U = len(ustart)
    NEG = -1e30
    d = np.full(S, NEG)
    rec = np.full(S, -1, dtype=np.int64)
    max_rec = T * U + U
    rec_unit = np.zeros(max_rec, dtype=np.int64)
    rec_prev = np.zeros(max_rec, dtype=np.int64)
    nrec = 0
    for v in range(U):
        s = ustart[v]
        d[s] = lmw * lm_start[v] + bonus[v] + ll[0, s]
        rec_unit[nrec] = v; rec_prev[nrec] = -1; rec[s] = nrec; nrec += 1
    nd = np.empty(S)
    nrecs = np.empty(S, dtype=np.int64)
    for t in range(1, T):
        # best exit of every unit
        best_v = np.full(U, NEG)
        best_u = np.full(U, -1, dtype=np.int64)
        for u in range(U):
            last = ustart[u] + ulen[u] - 1
            ex = d[last] + next_lp[last]
            if ex <= NEG / 2:
                continue
            for v in range(U):
                sc = ex + lmw * lm[u, v] + bonus[v]
                if sc > best_v[v]:
                    best_v[v] = sc
                    best_u[v] = u
        for u in range(U):
            for k in range(ulen[u]):
                s = ustart[u] + k
                stay = d[s] + self_lp[s]
                if k > 0:
                    move = d[s - 1] + next_lp[s - 1]
                    if move > stay:
                        nd[s] = move; nrecs[s] = rec[s - 1]
                    else:
                        nd[s] = stay; nrecs[s] = rec[s]
                else:
                    if best_v[u] > stay:
                        nd[s] = best_v[u]
                        lastp = ustart[best_u[u]] + ulen[best_u[u]] - 1
                        rec_unit[nrec] = u; rec_prev[nrec] = rec[lastp]; nrecs[s] = nrec; nrec += 1
                    else:
                        nd[s] = stay; nrecs[s] = rec[s]
        for s in range(S):
            d[s] = nd[s] + ll[t, s]
            rec[s] = nrecs[s]
    best, br = NEG, -1
    for u in range(U):
        last = ustart[u] + ulen[u] - 1
        sc = d[last] + lmw * lm_end[u]
        if sc > best:
            best, br = sc, rec[last]
    out = []
    while br >= 0:
        out.append(rec_unit[br])
        br = rec_prev[br]
    return out[::-1]


# ---------------------------------------------------------------- training
def state_seq(text, uidx, ustart, ulen):
    seq = []
    for sym in symbols(text):
        u = uidx[sym]
        seq += list(range(ustart[u], ustart[u] + ulen[u]))
    return np.array(seq, dtype=np.int64)


def bigram(texts, uidx, k=0.1):
    U = len(uidx)
    c = np.full((U, U), k); cs = np.full(U, k); ce = np.full(U, k)
    for t in texts:
        s = [uidx[x] for x in symbols(t)]
        cs[s[0]] += 1; ce[s[-1]] += 1
        for a, b in zip(s, s[1:]):
            c[a, b] += 1
    tot = c.sum(1) + ce
    return np.log(c / tot[:, None]), np.log(cs / cs.sum()), np.log(ce / tot)


def train():
    s = set_all_seeds()
    rng = np.random.default_rng(s)
    EXP.mkdir(parents=True, exist_ok=True)
    df = read_split("train")
    texts = [normalize(t) for t in df.text]
    units, ulen, ustart = units_from(texts)
    uidx = {u: i for i, u in enumerate(units)}
    S = int(ulen.sum())
    t0 = time.time()
    feats = features(df)
    print(f"features: {sum(len(f) for f in feats.values())} frames, {time.time() - t0:.0f}s", flush=True)
    seqs = {u: state_seq(t, uidx, ustart, ulen) for u, t in zip(df.utt_id, texts)}
    # flat start: uniform segmentation
    ali = {}
    for u, f in feats.items():
        q = seqs[u]
        ali[u] = q[np.minimum((np.arange(len(f)) * len(q)) // len(f), len(q) - 1)]
    self_lp = np.full(S, np.log(0.6)); next_lp = np.full(S, np.log(0.4))
    params = [None] * S
    for it, m in enumerate(MIX_SCHEDULE):
        X = np.concatenate([feats[u] for u in ali]); A = np.concatenate([ali[u] for u in ali])
        order = np.argsort(A, kind="stable"); A, X = A[order], X[order]
        bounds = np.searchsorted(A, np.arange(S + 1))
        for st in range(S):
            xs = X[bounds[st]:bounds[st + 1]]
            if len(xs) > MAX_FRAMES_PER_STATE:
                xs = xs[rng.choice(len(xs), MAX_FRAMES_PER_STATE, replace=False)]
            params[st] = fit_state(xs.astype(np.float64), m, params[st], rng)
        M = max(p[0].shape[0] for p in params)
        means = np.zeros((S, M, 39)); var = np.ones((S, M, 39)); logw = np.full((S, M), -1e10)
        for st, (mu, v, w) in enumerate(params):
            k = len(w)
            means[st, :k], var[st, :k], logw[st, :k] = mu, v, np.log(w + 1e-12)
        gmm = GMM(means.astype(np.float32), var.astype(np.float32), logw.astype(np.float32))
        # transitions from current alignment: P(leave) = 1 / mean duration
        occ = np.bincount(A, minlength=S).astype(float)
        visits = np.zeros(S)
        for u in ali:
            a = ali[u]
            visits += np.bincount(a[np.r_[True, a[1:] != a[:-1]]], minlength=S)
        p_leave = np.clip(visits / np.maximum(occ, 1), 0.05, 0.95)
        self_lp, next_lp = np.log(1 - p_leave), np.log(p_leave)
        # Viterbi re-alignment
        tot, failed = 0.0, 0
        for u, f in feats.items():
            path, ok = align(gmm.loglik(f), seqs[u], self_lp, next_lp)
            if ok:
                ali[u] = path
            else:
                failed += 1
        print(f"pass {it}: {m} gauss/state, failed alignments {failed}, {time.time() - t0:.0f}s", flush=True)
    lm, lm_s, lm_e = bigram(texts, uidx)
    np.savez(EXP / "model.npz", means=gmm.means, var=gmm.var, logw=gmm.logw, self_lp=self_lp,
             next_lp=next_lp, ulen=ulen, ustart=ustart, lm=lm, lm_s=lm_s, lm_e=lm_e,
             units=np.array(units), train_seconds=time.time() - t0)


# ---------------------------------------------------------------- decoding
def to_text(units, seq):
    s = "".join(" " if units[u] == SP else ("" if units[u] == SIL else units[u]) for u in seq)
    return " ".join(s.split())


def decode(split):
    s = set_all_seeds()
    z = np.load(EXP / "model.npz")
    gmm = GMM(z["means"], z["var"], z["logw"])
    units = list(z["units"])
    df = read_split(split)
    feats = features(df)
    t0 = time.time()
    lls = {u: gmm.loglik(f) for u, f in feats.items()}

    def run(lmw, wip, wb):
        bonus = np.full(len(units), wip)
        bonus[units.index(SP)] += wb
        return [to_text(units, decode_loop(lls[u], z["ustart"], z["ulen"], z["self_lp"], z["next_lp"],
                                           z["lm"], z["lm_s"], z["lm_e"], lmw, bonus)) for u in df.utt_id]

    choice_file = EXP / "dev_choice.json"
    if split == "dev":
        grid = []
        refs = [normalize(t) for t in df.text]
        for lmw in (4.0, 6.0, 8.0, 10.0):
            for wip in (0.0, 2.0, 4.0, 6.0, 8.0):
                for wb in (0.0, 4.0, 8.0, 12.0, 16.0):
                    sc = corpus_scores(refs, run(lmw, wip, wb))
                    grid.append({"lm_weight": lmw, "insertion_penalty": wip, "word_bonus": wb,
                                 "dev_wer": sc["wer"], "dev_cer": sc["cer"]})
                print(grid[-1], flush=True)
        best = min(grid, key=lambda g: g["dev_wer"])
        choice_file.write_text(json.dumps({"best": best, "grid": grid}, indent=2), encoding="utf-8")
    best = json.loads(choice_file.read_text(encoding="utf-8"))["best"]
    t1 = time.time()
    hyps = run(best["lm_weight"], best["insertion_penalty"], best["word_bonus"])
    compute = (t1 - t0 if split == "test" else 0) + time.time() - t1  # dev: grid search not counted
    run_info = dict(
        model_id="ours: MFCC+GMM-HMM (graphemes)", revision="n/a (trained from scratch)",
        family="B1 classical MFCC + GMM-HMM", trained_by_us=True, seed=s,
        decoding=f"Viterbi grapheme loop, char bigram LM, lm_weight={best['lm_weight']}, "
                 f"unit_insertion_bonus={best['insertion_penalty']}, word_boundary_bonus={best['word_bonus']} (chosen on dev)",
        features="13 MFCC + d + dd (39), per-speaker CMVN",
        toolkit="own Python implementation (numpy/numba/torch)",
        train_hours=round(read_split("train").duration.sum() / 3600, 2),
        train_seconds=float(z["train_seconds"]), gaussians_per_state=MIX_SCHEDULE[-1],
        selected="LM weight / insertion penalty by dev WER (grid in exp, copied to run info)",
        dev_grid=json.loads(choice_file.read_text(encoding="utf-8"))["grid"],
        compute_seconds=round(compute, 2))
    write_predictions("gmm_hmm", split, list(df.utt_id), hyps, run_info)
    print("done", split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "decode"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    train() if a.stage == "train" else decode(a.split)
