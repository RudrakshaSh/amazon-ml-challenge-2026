"""``ber decide`` (tasks T3.5/T4.1): pair scores -> final matches, scored on val.

1. one owner per query: each S2/S3 record keeps only its highest-scoring S1
   (EDA: every S2/S3 record matches at most one S1); optionally exact duplicate
   records inside a source are tied to one owner;
2. per S1 either
   * ``threshold``: keep assigned queries with p >= t, or
   * ``expected_f``: keep the prefix maximising expected F0.5 (see
     :mod:`ber.decide.expected_f`), after dropping p < ``min_p``;
3. val: macro F0.5 over every val S1 (singletons included), threshold swept;
   test: decisions written for ``ber submit``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..eval.metrics import per_entity, summarize
from ..eval.split import slice_mask
from ..log import timed
from . import anchor
from .expected_f import select

ANCHOR_COLS = ("hn_rel", "a_tsr")  # stage-2 score columns the anchor correction needs


def one_owner(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep each query's best S1 (ties: first)."""
    s = scores.sort_values(["q_id", "p"], ascending=[True, False])
    return s.drop_duplicates("q_id", keep="first")


def dup_keys(recs: pd.DataFrame) -> pd.Series:
    """Key shared by exact duplicate records (same source, country, raw name and
    address), indexed by entity_id."""
    key = recs.source.astype(str) + "|" + recs.country + "|" + recs.name_raw + "|" + recs.addr_raw
    return pd.Series(pd.util.hash_pandas_object(key, index=False).to_numpy(),
                     index=recs.entity_id.to_numpy())


def dup_keys_streamed(path, filters=None) -> pd.Series:
    """:func:`dup_keys` for the S2/S3 records of a ``records_<split>`` directory,
    one parquet part at a time (only the 8-byte hashes are kept, not the raw
    names and addresses)."""
    cols = ["entity_id", "source", "country", "name_raw", "addr_raw"]
    out = []
    for f in sorted(path.glob("*.parquet")):
        r = pd.read_parquet(f, columns=cols, filters=filters)
        r = r[r.source != 1]
        if len(r):
            out.append(dup_keys(r))
    return pd.concat(out) if out else pd.Series(dtype=np.uint64)


def tie_duplicates(assigned: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Exact duplicates within a source (~20k per source, EDA 01) must share an
    owner: every member takes the owner and p of the group's best-scoring member."""
    a = assigned.copy()
    a["k"] = keys.reindex(a.q_id.to_numpy()).to_numpy()
    grp = a.groupby("k")
    size = grp.q_id.transform("size")
    if (size > 1).any():
        best = a.loc[grp.p.idxmax(), ["k", "s1_id", "p"]].set_index("k")
        m = size > 1
        a.loc[m, "s1_id"] = best.s1_id.reindex(a.k[m]).to_numpy()
        a.loc[m, "p"] = best.p.reindex(a.k[m]).to_numpy()
    return a.drop(columns="k")


def decide(assigned: pd.DataFrame, method: str, threshold: float, min_p: float,
           empty_prior: float) -> pd.DataFrame:
    if method == "threshold":
        return assigned[assigned.p >= threshold]
    if method in ("expected_f", "expected_f_exact"):
        return select(assigned[assigned.p >= min_p], empty_prior,
                      exact=method == "expected_f_exact")
    raise ValueError(f"unknown decide.method {method!r}")


def _fillers(st: StageRun) -> frozenset[str]:
    with open(st.input_path("features", "fillers"), encoding="utf-8") as f:
        return frozenset(json.load(f))


def _kinds(a: pd.DataFrame, recs: pd.DataFrame, fillers: frozenset[str]) -> np.ndarray:
    """Name-change kind of each assigned pair; ``recs`` = one country's S1 (all of
    them: the token frequencies) and the queries (entity_id, source, name_core)."""
    core = recs.set_index("entity_id").name_core
    s1c = recs.name_core[(recs.source == 1).to_numpy()]
    return anchor.name_kinds(core.reindex(a.q_id).fillna("").to_numpy(),
                             core.reindex(a.s1_id).fillna("").to_numpy(),
                             anchor.s1_token_df(s1c), len(s1c), fillers)


def _anchor_val(st: StageRun, cfg: Config, assigned: pd.DataFrame, qc: pd.Series,
                chosen: dict) -> pd.DataFrame:
    """Val side of the anchor correction: per-country kind ratios (the model is
    calibrated on val, so they measure the estimator's bias), pooled into the bias
    factors that the test pass divides out. With ``prior_shift=anchor`` the correction
    is also applied to val (a no-op unless a kind's ratio moves away from the pooled
    bias by more than min_ratio)."""
    dc = cfg.decide
    fillers = _fillers(st)
    rec_path = st.input_path("normalize", "records_train")
    per_country, kinds = {}, {}
    country = qc.reindex(assigned.q_id).to_numpy()
    for c in sorted(pd.unique(country[pd.notna(country)])):
        m = country == c
        a = assigned[m]
        recs = pd.read_parquet(rec_path, columns=["entity_id", "source", "name_core", "side"],
                               filters=[("country", "==", c)])
        recs = recs[(recs.source == 1).to_numpy() | (recs.side == "val").to_numpy()]
        k = _kinds(a, recs, fillers)
        street = a.a_tsr.to_numpy() >= anchor.SAME_STREET
        per_country[str(c)] = anchor.kind_ratios(k[street], a.hn_rel.fillna(0).to_numpy()[street],
                                                 a.p.to_numpy()[street])
        kinds[str(c)] = (m, k)
        del recs
    bias = anchor.pooled_bias(per_country)
    chosen["anchor"] = {"val": per_country, "bias": bias}
    st.metric("anchor_val", per_country)
    st.metric("anchor_bias", bias)
    if dc.prior_shift != "anchor":
        return assigned
    out, reports = assigned.copy(), {}
    for c, (m, k) in kinds.items():
        adj, reports[c] = anchor.correct(assigned[m], k, bias, dc.anchor_min_ratio,
                                         dc.anchor_min_pairs, dc.anchor_max_ratio)
        out.loc[m, "p"] = adj.p.to_numpy()
    st.metric("anchor_val_applied", {c: {k: v["r_applied"] for k, v in r.items()}
                                     for c, r in reports.items()})
    return out


def _as_lists(kept: pd.DataFrame) -> dict[str, list[str]]:
    return kept.groupby("s1_id").q_id.apply(list).to_dict()


def evaluate(kept: pd.DataFrame, truth: dict[str, list[str]], s1_meta: pd.DataFrame) -> pd.DataFrame:
    """Per-entity F0.5 for every val S1, with country / has-Indic-match columns."""
    df = per_entity(truth, _as_lists(kept), keys=s1_meta.index)
    return df.join(s1_meta, on="s1_id")


def _decide_val(st: StageRun, cfg: Config) -> dict:
    """Val side: sweep the decision parameters, score macro F0.5, return the
    chosen parameters (also written as ``decision_params.json``)."""
    dc = cfg.decide
    # val side only: val S1, their true matches and every val-side query (memory)
    rec_path = st.input_path("normalize", "records_train")
    recs = pd.read_parquet(rec_path, columns=["entity_id", "source", "country", "side", "dev",
                                              "owner", "name_indic"],
                           filters=[("side", "==", "val")])
    s1 = recs[(recs.source == 1) & slice_mask(recs, cfg, "val")]
    s1_meta = s1.set_index("entity_id")[["country"]]
    q = recs[(recs.source != 1) & recs.owner.isin(s1_meta.index)]
    truth = q.groupby("owner").entity_id.apply(list).to_dict()
    indic_s1 = set(q.owner[q.name_indic])
    s1_meta["has_indic_match"] = s1_meta.index.isin(indic_s1)
    s1_meta["segment"] = np.where(s1_meta.index.isin(set(truth)), "has_matches", "singleton")
    st.metric("val_s1", {"n": len(s1_meta), "singletons": int((s1_meta.segment ==
                                                                "singleton").sum())})

    scores = st.read_input(dc.scores_stage, "scores_val").astype({"q_id": str, "s1_id": str})
    assigned = one_owner(scores)
    st.metric("val_one_owner_precision_at_1", round(float(assigned.y.mean()), 5))
    chosen_extra: dict = {}
    if all(c in assigned for c in ANCHOR_COLS):  # anchor bias factors (see decide/anchor.py)
        with timed(st.log, "anchor: val kind ratios"):
            assigned = _anchor_val(st, cfg, assigned,
                                   recs.set_index("entity_id").country, chosen_extra)
    elif dc.prior_shift == "anchor":
        raise ValueError("decide.prior_shift=anchor needs hn_rel/a_tsr in the scores "
                         "(decide.scores_stage=rerank)")
    if dc.tie_duplicates:
        q_keys = dup_keys_streamed(rec_path, filters=[("side", "==", "val")])
        before = assigned.s1_id.copy()
        assigned = tie_duplicates(assigned, q_keys)
        st.metric("val_duplicates_retied", int((assigned.s1_id != before).sum()))

    # sweep: threshold method over t, expected-F methods over min_p
    exp_method = dc.method if dc.method.startswith("expected_f") else "expected_f"
    sweep = {}
    for t in dc.sweep:
        f_thr = evaluate(decide(assigned, "threshold", t, 0, 1), truth, s1_meta).f05.mean()
        f_exp = evaluate(decide(assigned, exp_method, 0, t, dc.empty_prior), truth,
                         s1_meta).f05.mean()
        sweep[t] = {"threshold": round(float(f_thr), 5), "expected_f_min_p": round(float(f_exp), 5)}
    st.metric("val_sweep", sweep)
    best_t = max(dc.sweep, key=lambda t: sweep[t]["threshold"])
    best_m = max(dc.sweep, key=lambda t: sweep[t]["expected_f_min_p"])
    # the other expected-F variant at its best min_p, for comparison
    other = "expected_f" if exp_method == "expected_f_exact" else "expected_f_exact"
    st.metric(f"val_{other}_at_best_min_p", round(float(evaluate(
        decide(assigned, other, 0, best_m, dc.empty_prior), truth, s1_meta).f05.mean()), 5))
    chosen = {"method": dc.method,
              "threshold": best_t if dc.threshold is None else dc.threshold,
              "min_p": best_m if dc.min_p is None else dc.min_p}
    st.metric("chosen", chosen)
    chosen.update(chosen_extra)
    st.write_json(chosen, "decision_params")

    kept = decide(assigned, dc.method, chosen["threshold"], chosen["min_p"], dc.empty_prior)
    ent = evaluate(kept, truth, s1_meta)
    st.write_parquet(ent, "val_per_entity")
    st.write_parquet(kept.reset_index(drop=True), "decisions_val")
    st.metric("val_f05", round(float(ent.f05.mean()), 5))
    for by in ("country", "segment", "has_indic_match"):
        s = summarize(ent, by).round(5)
        st.metric(f"val_by_{by}", {str(k): v for k, v in s.to_dict("index").items()})
    st.metric("val_overall", summarize(ent).round(5).to_dict("index")["ALL"])

    return chosen


def run_decide(cfg: Config) -> None:
    dc = cfg.decide
    with StageRun(cfg, "decide") as st:
        if cfg.run_train:
            chosen = _decide_val(st, cfg)
        else:  # test-only pass: parameters chosen on val by the train pass
            with open(st.path("decision_params.json"), encoding="utf-8") as f:
                chosen = json.load(f)
        if cfg.run_test:
            with timed(st.log, "decide test"):
                # stream: every scores part holds complete queries, so the best row
                # per query can be taken part by part (~10M rows kept, not ~250M)
                src = st.input_path(dc.scores_stage, "scores_test")
                if dc.prior_shift not in ("none", "anchor"):
                    raise ValueError(f"unknown decide.prior_shift {dc.prior_shift!r}")
                s1_country = None
                rec_test = st.input_path("normalize", "records_test")
                if dc.prior_shift == "anchor" and "anchor" not in chosen:
                    raise ValueError("decide.prior_shift=anchor needs the val bias factors: "
                                     "re-run the train pass of `ber decide` on rerank scores")
                if dc.prior_shift == "anchor":
                    rt = pd.read_parquet(rec_test, columns=["entity_id", "country"],
                                         filters=[("source", "==", 1)])
                    s1_country = rt.set_index("entity_id").country
                fillers = _fillers(st) if dc.prior_shift == "anchor" else None
                parts, reports = [], {}
                for f in sorted(src.glob("*.parquet")):
                    d = pd.read_parquet(f).astype({"q_id": str, "s1_id": str})
                    c = str(s1_country[d.s1_id.iloc[0]]) if s1_country is not None and len(d) else ""
                    a = one_owner(d)
                    del d
                    if dc.prior_shift == "anchor" and len(a):
                        # one score part = one country (rerank writes part-<country>)
                        recs = pd.read_parquet(rec_test, columns=["entity_id", "source", "name_core"],
                                               filters=[("country", "==", c)])
                        k = _kinds(a, recs, fillers)
                        del recs
                        a, reports[f"{c}:{f.name}"] = anchor.correct(
                            a, k, chosen["anchor"]["bias"], dc.anchor_min_ratio,
                            dc.anchor_min_pairs, dc.anchor_max_ratio)
                    parts.append(a)
                if reports:
                    st.write_json(reports, "prior_shift_test")
                    st.metric("prior_shift_test", reports)
                a_t = pd.concat(parts, ignore_index=True)
                del parts
                if a_t.q_id.duplicated().any():  # a query split across parts: resolve globally
                    a_t = one_owner(a_t)
                if dc.tie_duplicates:
                    a_t = tie_duplicates(a_t, dup_keys_streamed(
                        st.input_path("normalize", "records_test")))
                kept_t = decide(a_t, dc.method, chosen["threshold"], chosen["min_p"],
                                dc.empty_prior)
                st.write_parquet(kept_t.reset_index(drop=True), "decisions_test")
                st.metric("test_matches", len(kept_t))
