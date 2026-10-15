"""Leaderboard metric: macro F0.5 over S1 entities (same rules as utils/score.py).

Per S1 entity:
  * truth empty,  prediction empty      -> 1.0
  * truth empty,  prediction non-empty  -> 0.0
  * truth non-empty, prediction empty   -> 0.0
  * otherwise F_0.5 = 1.25 * P * R / (0.25 * P + R)

``per_entity`` returns one row per S1 so results can be sliced by country,
script, cluster size, and so on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

BETA2 = 0.25


def f05(tp: int, n_pred: int, n_true: int) -> float:
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity(truth: Mapping[str, Iterable[str]], pred: Mapping[str, Iterable[str]],
               keys: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per S1 in ``keys`` (default: all truth keys); missing predictions
    count as empty."""
    rows = []
    for k in (truth.keys() if keys is None else keys):
        t, p = set(truth.get(k, ())), set(pred.get(k, ()))
        tp = len(t & p)
        rows.append((k, len(t), len(p), tp, f05(tp, len(p), len(t))))
    return pd.DataFrame(rows, columns=["s1_id", "n_true", "n_pred", "tp", "f05"])


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Macro F0.5 plus pair-level (micro) precision/recall, optionally grouped."""
    def agg(g: pd.DataFrame) -> pd.Series:
        tp, npred, ntrue = g.tp.sum(), g.n_pred.sum(), g.n_true.sum()
        single = g.n_true == 0
        return pd.Series({
            "n": len(g), "f05": g.f05.mean(),
            "pair_p": tp / npred if npred else 1.0, "pair_r": tp / ntrue if ntrue else 1.0,
            "singleton_frac": single.mean(),
            "singleton_acc": (g.n_pred[single] == 0).mean() if single.any() else float("nan"),
        })
    if by is None:
        return agg(df).to_frame("ALL").T
    return df.groupby(by).apply(agg, include_groups=False)


def macro_f05(truth: Mapping[str, Iterable[str]], pred: Mapping[str, Iterable[str]],
              keys: Iterable[str] | None = None) -> float:
    return float(per_entity(truth, pred, keys).f05.mean())
