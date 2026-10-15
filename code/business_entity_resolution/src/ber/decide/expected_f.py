"""Per-S1 expected-F0.5 subset selection (task T4.1).

For one S1 with candidate queries of match probability p_1 >= p_2 >= ... the
prediction is always a prefix (top-k). For k > 0 the expected F0.5 is
approximated by the ratio of expectations

    E[F0.5 | top-k]  ~  1.25 * sum(p_1..p_k) / (0.25 * sum(p_all) + k)

(sum(p_all) = expected number of true matches among the candidates), and for
k = 0 ("no match") it is the probability that the S1 has no true match at all,
prod(1 - p_i), times ``p_empty_prior`` for matches that no candidate
covers. The k with the highest value wins.

``best_prefix_exact`` (v2, ``decide.method = expected_f_exact``) computes the
expectation exactly instead, treating the candidates as independent Bernoullis:
TP ~ PoissonBinomial(p_1..p_k), the missed ones M ~ PoissonBinomial(p_k+1..p_n),
E[F0.5] = sum P(TP=t) P(M=m) * 1.25 t / (0.25 (t + m) + k). The ratio of
expectations is biased for the small groups that dominate here (1-6
candidates per S1), e.g. it undervalues keeping a lone p=0.5 candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25


def best_prefix(p: np.ndarray, empty_prior: float = 1.0) -> tuple[int, float]:
    """(k, expected F0.5) for probabilities sorted in descending order."""
    if len(p) == 0:
        return 0, empty_prior
    csum = np.cumsum(p)
    total = csum[-1]
    k = np.arange(1, len(p) + 1)
    ef = (1 + BETA2) * csum / (BETA2 * total + k)
    e0 = float(np.prod(1.0 - p)) * empty_prior
    i = int(np.argmax(ef))
    return (0, e0) if e0 >= ef[i] else (i + 1, float(ef[i]))


def _poisson_binomial(p: np.ndarray) -> np.ndarray:
    """P(sum = j), j = 0..len(p), for independent Bernoulli(p_i)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for i, pi in enumerate(p):
        dist[1:i + 2] = dist[1:i + 2] * (1 - pi) + dist[0:i + 1] * pi
        dist[0] *= 1 - pi
    return dist


def best_prefix_exact(p: np.ndarray, empty_prior: float = 1.0) -> tuple[int, float]:
    """Like :func:`best_prefix` but with the exact expected F0.5 of each prefix."""
    n = len(p)
    if n == 0:
        return 0, empty_prior
    best_k, best_v = 0, float(np.prod(1.0 - p)) * empty_prior
    t = np.arange(n + 1)
    for k in range(1, n + 1):
        ptp = _poisson_binomial(p[:k])            # t = 0..k
        pm = _poisson_binomial(p[k:])             # m = 0..n-k
        tt, mm = t[:k + 1, None], t[None, :n - k + 1]
        f = (1 + BETA2) * tt / (BETA2 * (tt + mm) + k)
        v = float(ptp @ f @ pm)
        if v > best_v:
            best_k, best_v = k, v
    return best_k, best_v


def select(assigned: pd.DataFrame, empty_prior: float = 1.0,
           exact: bool = False) -> pd.DataFrame:
    """assigned: one row per query (s1_id, q_id, p), after one-owner assignment.
    Returns the kept rows."""
    prefix = best_prefix_exact if exact else best_prefix
    a = assigned.sort_values(["s1_id", "p"], ascending=[True, False])
    s1 = a.s1_id.to_numpy()
    p = a.p.to_numpy(dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, len(s1)]
    keep = np.zeros(len(a), dtype=bool)
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        k, _ = prefix(p[lo:hi], empty_prior)
        keep[lo:lo + k] = True
    return a[keep]
