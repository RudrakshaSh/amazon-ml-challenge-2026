"""Address normalisation (task T1.2).

Per comma-separated component:
  * placeholder components (null, <NULL>, N/A, ...) are dropped;
  * region components (state code/name, Indic-script state, French région or
    département) are removed and reported as ``region``;
  * the rest is folded and tokenised, street types / ordinals are canonicalised
    per country, unit/house prefixes (plot, flat, door, unit, #, N°) are dropped
    and zero padding is removed ("00132" -> "132").

``numbers`` keeps the raw digit strings (with padding) in order, for the
number-relation features (truncated / padded / shifted, see eda/FINDINGS.md §3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .geo import match_component, pick_region
from .lexicon import addr_junk, street_types
from .text import INDIC_RE, fold, tokens

PLACEHOLDERS = {"null", "<null>", "n/a", "na", "none", "nan", "-", "--"}
_DIGITS = re.compile(r"\d+")
_BOX = re.compile(r"\b(?:p\.?\s*o\.?\s*box|pmb)\b")


@dataclass(frozen=True)
class AddrNorm:
    norm: str          # normalised tokens in original order (region removed)
    bag: str           # sorted unique tokens (order-free form for n-gram indexing)
    region: str        # detected region code, "" if none
    numbers: str       # raw digit strings in order, space-separated
    has_box: bool      # PO box / PMB present (seen in lookalike distractors)
    n_placeholders: int


class AddressNormalizer:
    """Callable ``(raw_address, country) -> AddrNorm``. ``translit`` (optional)
    romanises stray Indic tokens other than state names."""

    def __init__(self, translit=None):
        self.translit = translit

    def __call__(self, raw: str, country: str) -> AddrNorm:
        street, junk = street_types(country), addr_junk(country)
        hits, toks, nums = [], [], []
        n_ph = 0
        for comp in raw.split(","):
            c = comp.strip()
            if not c:
                continue
            if c.lower() in PLACEHOLDERS:
                n_ph += 1
                continue
            hit = match_component(c, country)
            if hit:
                hits.append(hit)
                continue
            nums.extend(_DIGITS.findall(c))
            for t in tokens(fold(c)):
                if self.translit is not None and INDIC_RE.search(t):
                    t = self.translit(t)[0]
                t = street.get(t, t)
                if t in junk:
                    continue
                if t.isdigit():
                    t = t.lstrip("0") or "0"
                toks.append(t)
        return AddrNorm(norm=" ".join(toks), bag=" ".join(sorted(set(toks))),
                        region=pick_region(hits), numbers=" ".join(nums),
                        has_box=bool(_BOX.search(raw.lower())), n_placeholders=n_ph)
