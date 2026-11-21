"""Evaluation tests: the lecture's worked examples plus normalizer properties."""
import pytest

from src.eval.metrics import corpus_scores, per_utt_errors
from src.normalize import normalize


def test_lecture_cat_example():
    s = corpus_scores(["the cat sat on the mat"], ["the cat sit on the"])
    assert (s["sub"], s["del"], s["ins"]) == (1, 1, 0)
    assert s["wer"] == pytest.approx(0.3333, abs=1e-4)
    assert s["cer"] == pytest.approx(0.2273, abs=1e-4)


def test_lecture_insertions_example():
    s = corpus_scores(["go"], ["please go now today"])
    assert s["wer"] == pytest.approx(3.0)
    assert s["mer"] == pytest.approx(0.75)
    assert s["wil"] == pytest.approx(0.75)


def test_corpus_wer_is_pooled_not_averaged():
    refs = ["a b", "a b c d e f g h"]
    hyps = ["a x", "a b c d e f g x"]
    s = corpus_scores(refs, hyps)
    # pooled: 2 errors / 10 words = 0.20 ; mean of clip WERs would be 0.3125
    assert s["wer"] == pytest.approx(0.20)


def test_identical_is_zero():
    s = corpus_scores(["ala ma kota"], ["ala ma kota"])
    assert s["wer"] == 0 and s["cer"] == 0


def test_empty_hypothesis_is_all_deletions():
    s = corpus_scores(["ala ma kota"], [""])
    assert s["del"] == 3 and s["wer"] == 1.0


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        corpus_scores([""], ["x"])


def test_per_utt_errors_sum_to_corpus():
    refs = ["ala ma kota", "kot ma alę", "dzień dobry państwu"]
    hyps = ["ala ma psa", "kot ma", "dzień dobry pani państwu"]
    rows = per_utt_errors(refs, hyps)
    s = corpus_scores(refs, hyps)
    assert sum(r[0] for r in rows) / sum(r[1] for r in rows) == pytest.approx(s["wer"])
    assert sum(r[2] for r in rows) / sum(r[3] for r in rows) == pytest.approx(s["cer"])


@pytest.mark.parametrize("text", [
    "Panie Marszałku! Wysoka Izbo, „to” jest 100% prawda…",
    "Zażółć gęślą jaźń — ŻÓŁW.",
    "  dwa   spacje\tTAB ",
    "Unia Europejska (UE) - polsko-niemiecka współpraca?",
])
def test_normalizer_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


def test_normalizer_polish_letters_and_punct():
    assert normalize("Zażółć GĘŚLĄ, jaźń!") == "zażółć gęślą jaźń"


def test_normalizer_nfc():
    decomposed = "ż"  # z + combining dot above -> ż
    assert normalize(decomposed) == "ż"


def test_normalizer_hyphen_and_dash():
    assert normalize("polsko-niemiecka – współpraca") == "polsko niemiecka współpraca"


def test_paired_bootstrap_identical_systems_and_sign():
    import numpy as np
    from scripts.bootstrap import paired_bootstrap
    e = np.array([[1, 5, 2, 20], [1, 4, 1, 15], [2, 6, 3, 25]], dtype=float)
    same = paired_bootstrap(e, e.copy(), 200, np.random.default_rng(0))
    assert same["wer"]["delta"] == 0 and same["wer"]["ci95"] == [0.0, 0.0]
    better = e.copy(); better[:, 0] = 0; better[:, 2] = 0
    res = paired_bootstrap(better, e, 200, np.random.default_rng(0))
    assert res["wer"]["delta"] < 0 and res["wer"]["ci95"][1] < 0 and res["wer"]["p_a_not_better"] == 0
