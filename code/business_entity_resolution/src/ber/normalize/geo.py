"""State / region detection from free-text addresses, using ``tables/regions.tsv``.

Addresses are comma-separated components in no fixed order
("IA, Iowa City, 1064 Newton Rd" / "1795 Westchester Drive, High Point, NC"),
so detection looks at whole components:

* ``kind=code`` variants (``TX``, ``MH``) match only an exact upper-case component,
  so words such as "In" or "Or" inside ordinary text are never read as states;
* ``kind=name`` variants (``Texas``, ``Orissa``, ``महाराष्ट्र``) match
  case- and accent-insensitively.

When several components match, a code wins over a name. Among names,
spellings that are also common city names (``Washington``) lose to any other
match, then the last match wins.
"""

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path

TABLES = Path(__file__).resolve().parent / "tables"
UNKNOWN = ""
# spellings that are also frequent city names in the same country
AMBIGUOUS_NAMES = {("US", "washington")}


def _key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip().rstrip("."))
    s = "".join(ch for ch in s if not 0x0300 <= ord(ch) <= 0x036F)  # Latin accents only
    return " ".join(s.casefold().split())


@lru_cache(maxsize=1)
def region_table() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """({(country, CODE): code}, {(country, name_key): code})"""
    codes: dict[tuple[str, str], str] = {}
    names: dict[tuple[str, str], str] = {}
    with open(TABLES / "regions.tsv", encoding="utf-8") as f:
        rows = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for r in rows:
            if r["kind"] == "code":
                codes[(r["country"], r["variant"])] = r["code"]
            else:
                names[(r["country"], _key(r["variant"]))] = r["code"]
    return codes, names


def match_component(comp: str, country: str) -> tuple[str, str, bool] | None:
    """If one comma-separated address component is a region, return
    ``(code, kind, ambiguous)``; otherwise None."""
    comp = comp.strip()
    if not comp:
        return None
    codes, names = region_table()
    c = codes.get((country, comp))
    if c:
        return c, "code", False
    k = _key(comp)
    c = names.get((country, k))
    if c:
        return c, "name", (country, k) in AMBIGUOUS_NAMES
    return None


def pick_region(hits: list[tuple[str, str, bool]]) -> str:
    """Choose among component matches: last code, else last unambiguous name,
    else last name."""
    codes = [c for c, kind, _ in hits if kind == "code"]
    if codes:
        return codes[-1]
    clear = [c for c, _, amb in hits if not amb]
    if clear:
        return clear[-1]
    return hits[-1][0] if hits else UNKNOWN


def detect_region(address: str, country: str) -> str:
    """Region code for an address (e.g. ``TX``, ``MH``, ``HDF``), or ``""``."""
    if not address:
        return UNKNOWN
    hits = [h for h in (match_component(c, country) for c in address.split(",")) if h]
    return pick_region(hits)
