"""Indic-script -> Latin transliteration (task T1.3).

Indic names in S2/S3 are phonetic spellings of English words
("Premier Fortune Agro" -> "प्रीमियर फॉर्च्यून एग्रो") in 9 scripts, and the same
words repeat heavily ("प्राइवेट लिमिटेड" = private limited). So:

1. ``learn_dictionary``: from train-side true pairs whose Indic name and S1 name
   have the same number of tokens, count positional (indic_token, latin_token)
   pairs and keep the dominant Latin spelling of each Indic token;
2. ``Transliterator``: dictionary lookup first; otherwise romanise by rules
   (indic-transliteration -> IAST -> ASCII, anusvara -> n, final-schwa deletion
   for the north-Indian scripts) and snap the result to a word of the S1 name
   vocabulary with the same consonant skeleton ("praivet" -> prvt -> "private").

The skeleton step matters for scripts/spellings absent from the training pairs
(e.g. Gurmukhi: Punjab is a val region). S1 is fully known at inference time
(test included), so its vocabulary is legitimate to use.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

import pandas as pd
from indic_transliteration import sanscript

from .lexicon import read_table
from .text import INDIC_RE, fold, tokens

SCRIPTS = [  # (first, last, indic_transliteration scheme)
    (0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI), (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA), (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU), (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
# scripts whose words drop the inherent final vowel in speech ("limiṭeḍa" -> "limited")
SCHWA_DELETING = {sanscript.DEVANAGARI, sanscript.BENGALI, sanscript.GURMUKHI,
                  sanscript.GUJARATI, sanscript.ORIYA}
_NOT_ASCII_ALNUM = re.compile(r"[^a-z0-9]+")
_FINAL_SCHWA = re.compile(r"(?<=[^aeiou])a$")
_SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"), ("bh", "b"),
              ("kh", "k"), ("gh", "g"), ("jh", "j"), ("z", "j"), ("q", "k"), ("x", "ks"),
              ("w", "v"), ("ck", "k"), ("c", "k")]
_VOWELS = re.compile(r"[aeiouy]")
_REPEATS = re.compile(r"(.)\1+")


def script_of(tok: str) -> str | None:
    for ch in tok:
        o = ord(ch)
        for lo, hi, scheme in SCRIPTS:
            if lo <= o <= hi:
                return scheme
    return None


def rule_translit(tok: str) -> str:
    """Rule-based romanisation of one Indic token to plain ASCII."""
    scheme = script_of(tok)
    if scheme is None:
        return tok
    iast = sanscript.transliterate(tok, scheme, sanscript.IAST)
    iast = iast.replace("ṃ", "n").replace("ṁ", "n")  # anusvara is /n/ before most consonants
    s = unicodedata.normalize("NFKD", iast)
    s = _NOT_ASCII_ALNUM.sub("", "".join(ch for ch in s if ord(ch) < 128).lower())
    if scheme in SCHWA_DELETING and len(s) > 3:
        s = _FINAL_SCHWA.sub("", s)
    return s


def skeleton(word: str) -> str:
    """Consonant skeleton: merge sound-alike spellings, drop vowels, collapse
    repeats; a final j/z becomes s (English plural spelled with ज़/ਜ਼)."""
    w = word.lower()
    for a, b in _SKEL_SUBS:
        w = w.replace(a, b)
    w = _REPEATS.sub(r"\1", _VOWELS.sub("", w))
    return w[:-1] + "s" if w.endswith("j") else w


def build_vocab(names: Iterable[str], min_count: int = 2) -> dict[str, str]:
    """skeleton -> most frequent Latin word with that skeleton, from S1 names."""
    counts = Counter(t for n in names for t in tokens(fold(n)) if t.isalpha() and t.isascii())
    vocab: dict[str, str] = {}
    for w, c in counts.most_common():
        if c < min_count:
            break
        vocab.setdefault(skeleton(w), w)
    return vocab


def overrides() -> dict[str, str]:
    return {r["romanised"]: r["latin"] for r in read_table("translit_overrides.tsv")}


class Transliterator:
    """Callable ``token -> (latin, resolved)``, memoised. ``resolved`` is False
    only when neither the dictionary nor the vocabulary knew the word."""

    def __init__(self, dictionary: dict[str, str] | None = None,
                 vocab: dict[str, str] | None = None):
        self.dictionary = dictionary or {}
        self.vocab = vocab or {}
        self.overrides = overrides()
        self._cache: dict[str, tuple[str, bool]] = {}

    def __call__(self, tok: str) -> tuple[str, bool]:
        hit = self._cache.get(tok)
        if hit is None:
            if tok in self.dictionary:
                hit = (self.dictionary[tok], True)
            else:
                r = rule_translit(tok)
                if r in self.overrides:
                    hit = (self.overrides[r], True)
                else:
                    w = self.vocab.get(skeleton(r)) if len(r) >= 3 else None
                    hit = (w, True) if w else (r, False)
            self._cache[tok] = hit
        return hit


def aligned_token_pairs(indic_names: Iterable[str], latin_names: Iterable[str]):
    """Yield (indic_token, latin_token) for name pairs with equal token counts."""
    for ind, lat in zip(indic_names, latin_names):
        ti, tl = tokens(fold(ind)), tokens(fold(lat))
        if len(ti) == len(tl):
            for a, b in zip(ti, tl):
                if INDIC_RE.search(a) and not INDIC_RE.search(b):
                    yield a, b


def learn_dictionary(indic_names: Iterable[str], latin_names: Iterable[str],
                     min_count: int = 2, min_share: float = 0.6) -> pd.DataFrame:
    """Dominant Latin spelling per Indic token. Returns columns
    indic, latin, count, share, n_total."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for a, b in aligned_token_pairs(indic_names, latin_names):
        counts[a][b] += 1
    rows = []
    for a, c in counts.items():
        latin, n = c.most_common(1)[0]
        total = sum(c.values())
        if n >= min_count and n / total >= min_share:
            rows.append((a, latin, n, n / total, total))
    return pd.DataFrame(rows, columns=["indic", "latin", "count", "share", "n_total"])


def dictionary_coverage(dictionary: dict[str, str], vocab: dict[str, str],
                        indic_names: Iterable[str], latin_names: Iterable[str]) -> dict[str, float]:
    """On aligned held-out pairs: share of Indic tokens in the dictionary, and
    share transliterated exactly to the S1 token by (a) the full transliterator,
    (b) dictionary + rules only, (c) rules only."""
    full, no_vocab = Transliterator(dictionary, vocab), Transliterator(dictionary)
    n = in_dict = exact = exact_dict_rules = exact_rules = 0
    for a, b in aligned_token_pairs(indic_names, latin_names):
        n += 1
        in_dict += a in dictionary
        exact += full(a)[0] == b
        exact_dict_rules += no_vocab(a)[0] == b
        exact_rules += rule_translit(a) == b
    d = max(n, 1)
    return {"aligned_tokens": n, "in_dictionary": in_dict / d, "exact_match": exact / d,
            "exact_match_dict_rules": exact_dict_rules / d, "exact_match_rules_only": exact_rules / d}
