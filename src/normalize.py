"""The ONE text-normalization policy, applied to references and hypotheses of every system.

Steps (order matters):
1. Unicode NFC (so "z" + combining dot and "ż" are the same string).
2. Lowercase (Python's str.lower is correct for Polish; no dotted/dotless-i issue).
3. Numbers: every digit run is written out as a Polish cardinal with num2words
   ("2015" -> "dwa tysiące piętnaście"). Our cleaned references contain no digits
   (clips whose original text had digits were removed in cleaning), so this only
   affects hypotheses: a system that writes "2015" is not punished for using digits.
   Cardinal form only: inflected/ordinal forms ("dwa tysiące piętnastego") still count
   as errors, which is the honest outcome for a word-level metric.
4. Letters with diacritics that are NOT part of the Polish alphabet (á, ö, ü, č, š, ñ ...)
   are reduced to their base letter. Polish letters ą ć ę ł ń ó ś ź ż are kept.
   Reason: foreign names ("Pöttering") are spelled inconsistently by systems.
5. Apostrophes are deleted ("hollande'a" -> "hollandea"); Polish uses them only in
   inflected foreign names.
6. Hyphens, dashes and all other punctuation become spaces.
7. Whitespace is collapsed.

We do NOT use Whisper's EnglishTextNormalizer (English-specific spelling rules).
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

POLISH_DIACRITICS = set("ąćęłńóśźż")
_DIGITS = re.compile(r"\d+")
_APOS = re.compile(r"['’ʼ`´ˮ]")
_NON_WORD = re.compile(r"[^\w\s]|_")


def _num(m: re.Match) -> str:
    return " " + num2words(int(m.group(0)), lang="pl") + " "


def _fold_foreign(ch: str) -> str:
    if ch in POLISH_DIACRITICS or ch.isascii():
        return ch
    base = unicodedata.normalize("NFD", ch)[0]
    return base if base.isascii() and base.isalpha() else ch


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = _DIGITS.sub(_num, t)
    t = "".join(_fold_foreign(c) for c in t)
    t = _APOS.sub("", t)
    t = _NON_WORD.sub(" ", t)
    return " ".join(t.split())


def orthographic(text: str) -> str:
    """Orthographic form: keep casing and punctuation, only NFC + whitespace collapse."""
    return " ".join(unicodedata.normalize("NFC", text).split())
