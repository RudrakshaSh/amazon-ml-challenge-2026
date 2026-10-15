"""Anchor-based prior correction per name-change kind (v2.3, ``decide.prior_shift = anchor``).

Why: France (15 % of the test S1, no labels anywhere) varies business names in a way
US and India never do: the generic word of a name is swapped for another generic word
at the same address ("Ski College SARL" -> "Ski Comite SARL", "BYH Club SAS" -> "BYH
Loisirs SAS", "Dessin Club" -> "Dessin & Fils"). In US/India val, one-word swaps
between common words are rare (0.1 % of same-street pairs in the US, 1.5 % in India)
and mostly decoys, so the model gives the ~72k French ones p ~ 0.4 and links about a
third of them. Whether they are true copies can be read off the unlabelled test data
from a second, independent signal: the house-number relation.

Given the match status, the house-number relation of a record and its S1 (``hn_rel``:
missing / equal / padded / truncated / shifted / different) barely depends on how the
name was changed: on val, the true copies' hn_rel profile is the same for exact names,
typos, generic-word swaps and other changes. So within a name-change kind k the hn_rel
histogram is a mixture ``h_k = pi_k h1 + (1 - pi_k) h0`` of the true-copy profile h1
and the decoy profile h0, and the share of true copies pi_k follows by least squares.
h1 and h0 are taken from the exact-name kind with the model's p as soft labels (on val
they match the true profiles to within 0.002). The model's own view of kind k is its
mean p there, so

    r_k = odds(pi_k) / odds(mean p over kind k)

is the factor by which the model misjudges the prior odds of that kind.

The estimator is biased for the non-exact kinds: decoys with an edited name keep their
house number more often than same-name decoys, so pi_k comes out too high there. The
bias is measured on val, where the model is calibrated and r_k should be 1: b_k =
r_k(val), pooled over the val countries (log-mean weighted by pairs), and divided out
on test. A kind is corrected only if the de-biased ratio r_k / b_k is beyond
``min_ratio`` (default 2; the largest val deviation of a single country from the pooled
bias is 1.7x) in either direction and it has >= ``min_pairs`` pairs; then every pair of
the kind gets ``p' = r p / (r p + 1 - p)``, with r capped at ``max_ratio``.

Only pairs on the same street as their S1 (address token-set ratio >= 80) take part:
elsewhere the house number says nothing about the match.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

KINDS = ("equal_key", "rare_swap", "generic_swap", "other")
N_HN = 6               # hn_rel codes 0..5 (features/numbers.py)
SAME_STREET = 80.0     # address token-set ratio for the house number to be comparable
GENERIC_DF = 1e-3      # a token in >= 0.1 % of the country's S1 names is generic


def s1_token_df(s1_cores) -> Counter:
    """Number of S1 names containing each core-name token."""
    return Counter(t for n in s1_cores for t in set(n.split()))


def name_kinds(q_core, s_core, df: Counter, n_s1: int, fillers: frozenset[str]) -> np.ndarray:
    """Kind of name change per pair, on the filler-free name keys (as the features
    stage builds them): 0 equal key, 1 one rare token swapped (typo-like), 2 one
    generic token swapped (both tokens in >= 0.1 % of the country's S1 names),
    3 anything else."""
    cache: dict[str, frozenset] = {}

    def key(n: str) -> frozenset:
        k = cache.get(n)
        if k is None:
            t = set(n.split())
            k = frozenset((t - fillers) or t)
            cache[n] = k
        return k

    thr = GENERIC_DF * max(n_s1, 1)
    out = np.empty(len(q_core), dtype=np.int8)
    for i, (a, b) in enumerate(zip(q_core, s_core)):
        ka, kb = key(a), key(b)
        qa, sb = ka - kb, kb - ka
        if not qa and not sb:
            out[i] = 0
        elif len(qa) == 1 and len(sb) == 1 and ka & kb:
            (x,), (y,) = qa, sb
            out[i] = 2 if min(df.get(x, 0), df.get(y, 0)) >= thr else 1
        else:
            out[i] = 3
    return out


def hn_hist(hn: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    h = np.bincount(np.clip(hn.astype(np.int64), 0, N_HN - 1), weights=w, minlength=N_HN)
    s = h.sum()
    return h / s if s > 0 else h.astype(np.float64)


def _odds(x: float) -> float:
    return x / (1.0 - x)


def kind_ratios(kind: np.ndarray, hn: np.ndarray, p: np.ndarray) -> dict[str, dict]:
    """Per kind (same-street pairs of one country): n, pi_anchor (true-copy share from
    the hn_rel mixture), pi_model (mean p) and r = odds(pi_anchor) / odds(pi_model)."""
    p = p.astype(np.float64)
    eq = kind == 0
    out: dict[str, dict] = {}
    if eq.sum() == 0:
        return out
    h1, h0 = hn_hist(hn[eq], p[eq]), hn_hist(hn[eq], 1.0 - p[eq])
    d = h1 - h0
    den = float(d @ d)
    for k, name in enumerate(KINDS):
        m = kind == k
        if not m.any() or den < 1e-9:
            continue
        pa = float(np.clip((hn_hist(hn[m]) - h0) @ d / den, 0.01, 0.99))
        pm = float(np.clip(p[m].mean(), 0.01, 0.99))
        out[name] = {"n": int(m.sum()), "pi_anchor": round(pa, 5), "pi_model": round(pm, 5),
                     "r": round(_odds(pa) / _odds(pm), 5)}
    return out


def pooled_bias(per_country: dict[str, dict[str, dict]]) -> dict[str, float]:
    """b_k: pair-weighted log-mean of the val ratios over countries."""
    out = {}
    for name in KINDS:
        rows = [v[name] for v in per_country.values() if name in v]
        n = sum(r["n"] for r in rows)
        if n:
            out[name] = round(float(np.exp(sum(r["n"] * np.log(r["r"]) for r in rows) / n)), 5)
    return out


def correct(assigned: pd.DataFrame, kind: np.ndarray, bias: dict[str, float],
            min_ratio: float = 2.0, min_pairs: int = 2000,
            max_ratio: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """Rescale p of the same-street pairs of every kind whose de-biased ratio is beyond
    ``min_ratio`` (one country's assigned pairs: q_id, s1_id, p, hn_rel, a_tsr).
    ``kind`` is aligned with ``assigned``. Returns the frame and a report per kind."""
    a = assigned.copy()
    street = (a.a_tsr.to_numpy() >= SAME_STREET)
    p = np.array(a.p.to_numpy(), dtype=np.float64, copy=True)  # (pandas may hand out a read-only view)
    hn = a.hn_rel.fillna(0).to_numpy()
    rat = kind_ratios(kind[street], hn[street], p[street])
    report = {}
    for k, name in enumerate(KINDS):
        if name not in rat:
            continue
        row = dict(rat[name])
        row["bias"] = bias.get(name, 1.0)
        r = row["r"] / row["bias"]
        row["r_debiased"] = round(r, 5)
        apply = row["n"] >= min_pairs and (r >= min_ratio or r <= 1.0 / min_ratio)
        if apply:
            r = float(np.clip(r, 1.0 / max_ratio, max_ratio))
            m = street & (kind == k)
            p[m] = r * p[m] / (r * p[m] + 1.0 - p[m])
        row["r_applied"] = round(r, 5) if apply else 1.0
        report[name] = row
    a["p"] = p.astype(np.float32)
    return a, report
