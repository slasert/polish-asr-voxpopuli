"""Corpus-level ASR metrics. The only place WER/CER are computed.

All rates are pooled over the corpus: sum(S+D+I) / sum(N). We never
average per-clip WERs (a 2-word clip would weigh as much as a 30-word clip).
"""
from __future__ import annotations

import jiwer


def _counts(refs, hyps, char=False):
    if char:
        out = jiwer.process_characters(list(refs), list(hyps))
    else:
        out = jiwer.process_words(list(refs), list(hyps))
    return out


def corpus_scores(refs, hyps):
    """Return a dict of pooled word- and character-level scores.

    refs/hyps: lists of strings of equal length. Empty references are not
    allowed (the cleaning step removes them); an empty hypothesis is fine.
    """
    refs, hyps = list(refs), list(hyps)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} refs vs {len(hyps)} hyps")
    if any(not r.strip() for r in refs):
        raise ValueError("empty reference")
    w = _counts(refs, hyps)
    c = _counts(refs, hyps, char=True)
    n_words = w.hits + w.substitutions + w.deletions
    n_chars = c.hits + c.substitutions + c.deletions
    return {
        "wer": w.wer,
        "mer": w.mer,
        "wil": w.wil,
        "cer": c.cer,
        "sub": w.substitutions,
        "del": w.deletions,
        "ins": w.insertions,
        "hits": w.hits,
        "n_ref_words": n_words,
        "char_sub": c.substitutions,
        "char_del": c.deletions,
        "char_ins": c.insertions,
        "n_ref_chars": n_chars,
        "n_utts": len(refs),
    }


def per_utt_errors(refs, hyps):
    """(word_errors, ref_words, char_errors, ref_chars) per utterance.

    Used by the paired bootstrap, which resamples utterances and re-pools.
    """
    rows = []
    for r, h in zip(refs, hyps):
        w = jiwer.process_words(r, h)
        c = jiwer.process_characters(r, h)
        rows.append((
            w.substitutions + w.deletions + w.insertions,
            w.hits + w.substitutions + w.deletions,
            c.substitutions + c.deletions + c.insertions,
            c.hits + c.substitutions + c.deletions,
        ))
    return rows
