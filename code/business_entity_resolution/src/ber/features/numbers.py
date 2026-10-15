"""Address-number relation features (task T3.1).

EDA (eda/FINDINGS.md §3): true pairs keep the house number, or **truncate** it
(7622 -> 762, 6665 -> 665), **zero-pad** it (2877 -> 002877) or drop it.
Lookalike distractors **shift** it by a small amount (805 -> 812, 5835 -> 5838).

v2 (``number_features_v2``): in true pairs the first number is often not the
house number at all. Indian S2/S3 addresses get a fake number *prepended*
("H.NO 937 33, Dr. Sarat Banerjee Road", "PLOT 57 1808/B4, ...") and components
are reordered ("Unit 177, AZ, 9310 Main Street" -> "AZ, 9310 MAIN ST"), so the
first numbers differ while the house number is still there. True pairs also
get single-digit typos in leading positions (394 -> 894, 5710 -> 4710), while
distractors change the last digits (5835 -> 5838). Hence: is either side's
first number anywhere in the other's numbers, the best relation of the S1's
first number to *any* query number, the digit edit distance, and the position
of the leftmost differing digit.
"""

from __future__ import annotations

import math

from rapidfuzz.distance import Levenshtein

# house-number relation codes (categorical feature)
MISSING, EQUAL, PADDED, TRUNCATED, SHIFTED, DIFFERENT = range(6)
SHIFT_MAX = 50  # |a - b| at or below this (same digit count) counts as a small shift


def relation(a: str, b: str) -> int:
    """Relation between two raw digit strings (e.g. house numbers)."""
    if not a or not b:
        return MISSING
    if a == b:
        return EQUAL
    sa, sb = a.lstrip("0") or "0", b.lstrip("0") or "0"
    if sa == sb:
        return PADDED
    if len(sa) != len(sb) and (sa.startswith(sb) or sb.startswith(sa) or
                               sa.endswith(sb) or sb.endswith(sa)):
        return TRUNCATED
    if len(sa) == len(sb) and abs(int(sa) - int(sb)) <= SHIFT_MAX:
        return SHIFTED
    return DIFFERENT


def number_features(q_nums: str, s_nums: str) -> tuple[int, float, int, int, int, float]:
    """Features for space-separated digit strings of the query and S1 address:
    (house-number relation, log1p |difference|, n common, n query-only,
    n S1-only, jaccard of the number sets)."""
    qn, sn = q_nums.split(), s_nums.split()
    rel = relation(qn[0] if qn else "", sn[0] if sn else "")
    diff = (math.log1p(abs(int(qn[0]) - int(sn[0]))) if qn and sn and
            len(qn[0]) < 12 and len(sn[0]) < 12 else -1.0)
    qs = {x.lstrip("0") or "0" for x in qn}
    ss = {x.lstrip("0") or "0" for x in sn}
    common = len(qs & ss)
    union = len(qs | ss)
    return rel, diff, common, len(qs - ss), len(ss - qs), (common / union if union else -1.0)


_PRIORITY = {EQUAL: 0, PADDED: 1, TRUNCATED: 2, SHIFTED: 3, DIFFERENT: 4}


def number_features_v2(q_nums: str, s_nums: str) -> tuple[int, int, int, int, int]:
    """(S1 first number in query numbers, query first number in S1 numbers,
    best relation of the S1 first number to any query number, digit edit
    distance of the first numbers, position from the right of the leftmost
    differing digit for equal-length first numbers). -1 when undefined."""
    qn = [x.lstrip("0") or "0" for x in q_nums.split()]
    sn = [x.lstrip("0") or "0" for x in s_nums.split()]
    if not qn or not sn:
        return -1, -1, MISSING, -1, -1
    s_in_q = int(sn[0] in qn)
    q_in_s = int(qn[0] in sn)
    best = min((relation(x, sn[0]) for x in qn), key=_PRIORITY.__getitem__)
    a, b = qn[0], sn[0]
    lev = Levenshtein.distance(a, b)
    pos = -1
    if len(a) == len(b) and a != b:
        pos = len(a) - 1 - next(i for i in range(len(a)) if a[i] != b[i])
    return s_in_q, q_in_s, best, lev, pos
