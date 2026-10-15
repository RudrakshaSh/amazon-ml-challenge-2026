"""Cached loaders for the lookup tables in ``normalize/tables/*.tsv``.

Tables are plain TSV with a header row; lines starting with ``#`` are comments.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

TABLES = Path(__file__).resolve().parent / "tables"


def read_table(name: str) -> list[dict[str, str]]:
    with open(TABLES / name, encoding="utf-8") as f:
        return list(csv.DictReader((line for line in f if not line.startswith("#")),
                                   delimiter="\t"))


@lru_cache(maxsize=None)
def legal_forms() -> dict[str, str]:
    return {r["variant"]: r["canonical"] for r in read_table("legal_forms.tsv")}


@lru_cache(maxsize=None)
def name_stopwords() -> frozenset[str]:
    return frozenset(r["token"] for r in read_table("name_stopwords.tsv"))


@lru_cache(maxsize=None)
def ocr_digit_map() -> dict[int, str]:
    return str.maketrans({r["digit"]: r["letter"] for r in read_table("ocr_digits.tsv")})


@lru_cache(maxsize=None)
def street_types(country: str) -> dict[str, str]:
    """Variant -> canonical for one country (country-specific rows override '*')."""
    rows = read_table("street_types.tsv")
    out = {r["variant"]: r["canonical"] for r in rows if r["country"] == "*"}
    out.update({r["variant"]: r["canonical"] for r in rows if r["country"] == country})
    return out


@lru_cache(maxsize=None)
def addr_junk(country: str) -> frozenset[str]:
    return frozenset(r["token"] for r in read_table("addr_junk.tsv")
                     if r["country"] in ("*", country))
