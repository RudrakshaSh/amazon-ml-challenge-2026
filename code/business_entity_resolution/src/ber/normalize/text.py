"""Text folding, tokenisation and business-name normalisation.

Name pipeline (``NameNormalizer.__call__``), each step driven by the EDA noise
catalogue (eda/FINDINGS.md §4):

1. fold: lowercase, strip Latin accents, ``&``/``+`` -> "and"
2. alias split: "<x> aka|fka|formerly|dba|doing business as|nee <y>" -> pre/alt parts
3. domain names: "www.unitedimpex.com" -> "unitedimpex" (flag ``is_domain``)
4. tokenise (keeping Indic letters and vowel signs), drop injected record ids and
   phone numbers ("#32038", "(ID: 63317)", "- 9334025793"), OCR digit fixes in mixed
   tokens ("c0mite" -> "comite"), join runs of single letters into known units or
   initials ("l l c" -> "llc", "a p" -> "ap", "d o m d" -> "do md")
5. transliterate Indic-script tokens to Latin (learned dictionary, then rules)
6. legal forms -> canonical and moved out of the core; stopwords dropped from the core
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz.distance import OSA

from .lexicon import legal_forms, name_stopwords, ocr_digit_map, read_table

INDIC_RE = re.compile("[\u0900-\u0DFF]")  # all 9 Indic scripts (letters + vowel signs)
_NON_WORD = re.compile("[^\\w\u0900-\u0DFF]+|_")
_ZW = re.compile("[\u200b-\u200d\ufeff]")  # zero-width (non-)joiners inside Indic words
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_HAS_DIGIT = re.compile(r"\d")
_HAS_ALPHA = re.compile(r"[a-z]")
_L = r"[\s./]*"
ALIAS_RE = re.compile(
    rf"\b(?:a{_L}k{_L}a|f{_L}k{_L}a|d{_L}b{_L}a|formerly(?:\s+known\s+as)?|"
    r"doing\s+business\s+as|also\s+known\s+as|known\s+as|nee)\b\s*:?")
DOMAIN_RE = re.compile(r"(?:\bwww\.)?\b([a-z0-9][a-z0-9-]*)\.(?:com|co\.in|in|net|org|fr|biz|info)\b")
# record ids the noise appends to S2/S3 names ("#32038"); see also _is_injected_number
HASH_NUM_RE = re.compile(r"#\s?\d{3,}")


def _is_injected_number(tok: str) -> bool:
    """Numbers the noise appends to S2/S3 names: 5-digit record ids ("(ID: 63317)")
    and 10-digit phone numbers ("... - 9334025793"). S1 names only hold numbers of
    <= 4 digits (house numbers, union locals), almost never 5 or >= 7 digits."""
    return tok.isdigit() and (len(tok) == 5 or len(tok) >= 7)


def fold(s: str) -> str:
    """Lowercase; strip Latin accents (Indic vowel signs are kept); &/+ -> and."""
    s = s.lower()
    if not s.isascii():
        s = unicodedata.normalize("NFKD", _ZW.sub("", s))
        s = "".join(ch for ch in s if not 0x0300 <= ord(ch) <= 0x036F)
    return s.replace("&", " and ").replace("+", " and ")


def tokens(folded: str) -> list[str]:
    return _NON_WORD.sub(" ", folded).split()


@lru_cache(maxsize=1)
def _units() -> frozenset[str]:
    return frozenset(legal_forms()) | frozenset(r["token"] for r in read_table("credentials.tsv"))


def _segment(run: str) -> list[str] | None:
    """Split a string of letters into known units (legal forms/credentials), or None."""
    units = _units()
    best: list[list[str] | None] = [None] * (len(run) + 1)
    best[0] = []
    for i in range(1, len(run) + 1):
        for j in range(max(0, i - 6), i):
            if best[j] is not None and run[j:i] in units:
                cand = best[j] + [run[j:i]]
                if best[i] is None or len(cand) < len(best[i]):
                    best[i] = cand
    return best[-1]


def join_single_letters(toks: list[str]) -> list[str]:
    """Merge runs of >=2 single Latin letters: into known units if the run can be
    segmented that way ("l l c" -> llc, "d o m d p c" -> do md pc), else into
    one initials token ("a p" -> ap)."""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            joined = "".join(run)
            out.extend(_segment(joined) or [joined])
        else:
            out.extend(run)
        run.clear()

    for t in toks:
        if len(t) == 1 and "a" <= t <= "z":
            run.append(t)
        else:
            flush()
            out.append(t)
    flush()
    return out


def ocr_fix(tok: str) -> str:
    """Undo scanner-style digit-for-letter swaps inside mixed tokens."""
    if _HAS_DIGIT.search(tok) and _HAS_ALPHA.search(tok) and not _ORDINAL.match(tok):
        return tok.translate(ocr_digit_map())
    return tok


@dataclass(frozen=True)
class NameNorm:
    norm: str          # all tokens after normalisation (legal forms canonical)
    core: str          # without legal forms and stopwords
    legal: str         # sorted unique canonical legal forms
    pre_core: str      # core of the part before an alias keyword ("" if no alias)
    alt_core: str      # core of the part after an alias keyword ("" if no alias)
    compact: str       # core without spaces (robust to domains / spacing)
    is_domain: bool
    has_alias: bool
    indic: bool        # raw name contained Indic script
    n_indic_oov: int   # Indic tokens not in the learned dictionary


class NameNormalizer:
    """Callable ``(raw_name) -> NameNorm``. ``translit`` maps an Indic token to
    ``(latin, in_dictionary)``; see :mod:`ber.normalize.translit`."""

    def __init__(self, translit=None):
        self.translit = translit
        self.legal = legal_forms()
        self.stop = name_stopwords()
        # long legal words tolerate one typo ("prlvate", "limted", "priavte"); short ones
        # must be exact
        self._long_legal = [v for v in self.legal if len(v) >= 6]
        self._fuzzy_cache: dict[str, str | None] = {}

    def _legal_variant(self, t: str) -> str | None:
        """The legal-form variant ``t`` spells (exactly or with one typo, an adjacent
        swap counting as one: "privtae", "prviate"), or None."""
        if t in self.legal:
            return t
        if len(t) < 6 or not t.isalpha():
            return None
        if t not in self._fuzzy_cache:
            self._fuzzy_cache[t] = next(
                (v for v in self._long_legal
                 if OSA.distance(t, v, score_cutoff=1) <= 1), None)
        return self._fuzzy_cache[t]

    def _toks(self, folded: str) -> tuple[list[str], int]:
        toks, oov = [], 0
        for t in tokens(HASH_NUM_RE.sub(" ", folded)):
            if _is_injected_number(t):
                continue
            if self.translit is not None and INDIC_RE.search(t):
                t, known = self.translit(t)
                oov += not known
                toks.extend(t.split())
            else:
                toks.append(ocr_fix(t))
        toks = join_single_letters(toks)
        # the noise doubles words ("Dynamic Dynamic Retail", "Of Of"); S1 never does
        return [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]], oov

    def _split(self, toks: list[str]) -> tuple[list[str], list[str], list[str]]:
        variants = [self._legal_variant(t) for t in toks]
        canon = [self.legal[v] if v else t for t, v in zip(toks, variants)]
        core = [c for c, v in zip(canon, variants) if not v and c not in self.stop]
        if not core:  # e.g. "The Company": keep something to compare
            core = [c for c, v in zip(canon, variants) if not v] or canon
        return canon, core, sorted({self.legal[v] for v in variants if v})

    def __call__(self, raw: str) -> NameNorm:
        f = fold(raw)
        domains = DOMAIN_RE.findall(f)
        if domains:
            f = DOMAIN_RE.sub(lambda m: f" {m.group(1)} ", f)
        pre = alt = ""
        m = ALIAS_RE.search(f)
        if m:
            pre, alt = f[:m.start()], f[m.end():]
            f = f"{pre} {alt}"
        toks, oov = self._toks(f)
        canon, core, legal = self._split(toks)
        pre_core = alt_core = ""
        if m:
            pre_core = " ".join(self._split(self._toks(pre)[0])[1]) if tokens(pre) else ""
            alt_core = " ".join(self._split(self._toks(alt)[0])[1]) if tokens(alt) else ""
        return NameNorm(
            norm=" ".join(canon), core=" ".join(core), legal=" ".join(legal),
            pre_core=pre_core, alt_core=alt_core, compact="".join(core),
            is_domain=bool(domains) and len(core) <= len(domains),
            has_alias=m is not None, indic=bool(INDIC_RE.search(raw)), n_indic_oov=oov)
