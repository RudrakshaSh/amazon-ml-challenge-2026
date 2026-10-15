"""``ber candidates`` (tasks T2.2/T2.3): per-country TF-IDF top-k candidates.

Each S2/S3 record (query) searches the S1 records of its country in three
character-trigram TF-IDF indexes:

  * ``name``:  normalised core name (Indic already transliterated)
  * ``addr``:  normalised address tokens, sorted (order-free)
  * ``combo``: name + address together (best single index for true pairs)

and keeps the union of the top-k of each. For every kept pair all three cosines
are computed, plus the pair's rank in each index (``k`` = "not in that top-k").
With ``block_by_region`` each query searches only the S1 records of its region's
neighbourhood (learned from train pairs, e.g. AP+TG); queries with no region,
or a region absent from S1, search the whole country. Queries with an empty
address take ``k_name_noaddr`` name candidates.

Outputs ``candidates_<split>.parquet``; for train, also a recall report.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..log import timed
from .tfidf import fit_index, gpu_available, pair_cosine, ranks_within_query, topk

INDEXES = ("name", "addr", "combo")
RECORD_COLS = ["entity_id", "source", "country", "name_core", "addr_bag", "region",
               "name_indic"]
TRAIN_COLS = ["side", "owner"]


def _texts(df: pd.DataFrame, index: str) -> list[str]:
    if index == "name":
        return df.name_core.tolist()
    if index == "addr":
        return df.addr_bag.tolist()
    return (df.name_core + " " + df.addr_bag).tolist()


def region_neighbors(recs: pd.DataFrame, min_share: float) -> dict[str, dict[str, list[str]]]:
    """Per country: query region -> S1 regions to search. Learned from matched
    train pairs: every owner region receiving >= ``min_share`` of a query
    region's true matches is included, symmetrised (e.g. AP <-> TG: Telangana
    addresses still written as "Andhra Pradesh")."""
    s1_region = recs.loc[recs.source == 1].set_index("entity_id").region
    m = recs[(recs.source != 1) & (recs.owner != "") & (recs.region != "")]
    m = m.assign(o=s1_region.reindex(m.owner).to_numpy())
    m = m[m.o.notna() & (m.o != "")]
    out: dict[str, dict[str, list[str]]] = {}
    for country, g in m.groupby("country"):
        share = pd.crosstab(g.region, g.o, normalize="index")
        nb: dict[str, set[str]] = {r: {r} for r in set(share.index) | set(share.columns)}
        for r, row in share.iterrows():
            for o, v in row.items():
                if v >= min_share:
                    nb[r].add(o)
                    nb.setdefault(o, {o}).add(r)
        out[country] = {r: sorted(v) for r, v in nb.items()}
    return out


def _groups(s1: pd.DataFrame, q: pd.DataFrame, by_region: bool,
            neighbors: dict[str, list[str]]):
    """(s1 row positions, query row positions) blocks to search within."""
    if not by_region:
        yield np.arange(len(s1)), np.arange(len(q))
        return
    s1_reg, q_reg = s1.region.to_numpy(), q.region.to_numpy()
    known = set(np.unique(s1_reg)) - {""}
    for r in sorted(known):
        q_pos = np.flatnonzero(q_reg == r)
        if len(q_pos):
            yield np.flatnonzero(np.isin(s1_reg, neighbors.get(r, [r]))), q_pos
    orphans = np.flatnonzero(~np.isin(q_reg, list(known)))
    if len(orphans):
        yield np.arange(len(s1)), orphans


def resolve_device(cfg: Config) -> str:
    """candidates.device: "cpu", "gpu", or "auto" (GPU when CuPy and a CUDA device
    are available)."""
    d = cfg.candidates.device
    if d == "auto":
        return "gpu" if gpu_available() else "cpu"
    if d == "gpu" and not gpu_available():
        raise RuntimeError("candidates.device=gpu but CuPy / a CUDA device is not available")
    return d


def _search_group(mats, s_pos, q_pos, no_addr, n_s1, cfg: Config, threads: int,
                  device: str = "cpu"):
    """Top-k union for one block: returns (keys, {index: (keys, ranks)})."""
    cc = cfg.candidates
    k = {"name": cc.k_name, "addr": cc.k_addr, "combo": cc.k_combo}
    rank_of = {}
    # the GPU wins on large blocks only (dense per-query work grows with the block's
    # S1 count; the CPU kernel's with overlapping postings); small blocks stay on CPU
    if device == "gpu" and len(s_pos) < cc.gpu_min_s1:
        device = "cpu"
    for ix in INDEXES:
        Xs, Xq = mats[ix]
        parts_k, parts_r = [], []
        # (query subset, k) runs: empty-address queries skip the address index
        # and get more name candidates
        # empty-address queries: only the name index (their combo text is the name)
        runs = [(q_pos[~no_addr[q_pos]], k[ix])]
        if ix == "name":
            runs.append((q_pos[no_addr[q_pos]], cc.k_name_noaddr))
        for qp, kk in runs:
            if len(qp) == 0:
                continue
            qi, sj, _ = topk(Xq[qp], Xs[s_pos], kk, cc.chunk_size, threads, cc.min_sim,
                             device, cc.gpu_mem_gb)
            parts_k.append(qp[qi].astype(np.int64) * n_s1 + s_pos[sj])
            parts_r.append(ranks_within_query(qi))
        rank_of[ix] = (np.concatenate(parts_k) if parts_k else np.array([], dtype=np.int64),
                       np.concatenate(parts_r) if parts_r else np.array([], dtype=np.int16))
    return np.unique(np.concatenate([v[0] for v in rank_of.values()])), rank_of


def generate_country(s1: pd.DataFrame, q: pd.DataFrame, cfg: Config, log,
                     neighbors: dict[str, list[str]] | None = None):
    """Yield candidate-pair frames for one country, one search block at a time
    (blocks partition the queries, so each query appears in exactly one frame).
    s1/q: record frames (RECORD_COLS). Queries with an empty address search only
    the name index, with ``k_name_noaddr`` candidates."""
    cc = cfg.candidates
    k = {"name": cc.k_name, "addr": cc.k_addr, "combo": cc.k_combo}
    no_addr = (q.addr_bag == "").to_numpy()
    threads = cc.n_threads or max(1, (os.cpu_count() or 4) - 2)
    device = resolve_device(cfg)
    log.info("  top-k search on %s", device)
    n_s1 = len(s1)
    fitted = {}
    for ix in INDEXES:
        vec, Xs = fit_index(_texts(s1, ix), cc.ngram, cc.max_df)
        fitted[ix] = (vec, Xs)
        log.info("  index %-5s: %d S1 x %d n-grams", ix, Xs.shape[0], Xs.shape[1])
    q_ids = pd.Categorical(q.entity_id)
    s_ids = pd.Categorical(s1.entity_id)
    for s_pos, q_pos in _groups(s1, q, cc.block_by_region, neighbors or {}):
        if len(s_pos) == 0 or len(q_pos) == 0:
            continue
        # query matrices are built per block (memory: one block's queries at a time);
        # positions below are local to the block
        qb = q.iloc[q_pos]
        mats = {ix: (Xs, vec.transform(_texts(qb, ix)).tocsr())
                for ix, (vec, Xs) in fitted.items()}
        local = np.arange(len(q_pos))
        keys, rank_of = _search_group(mats, s_pos, local, no_addr[q_pos], n_s1, cfg, threads,
                                      device)
        qi, sj = keys // n_s1, keys % n_s1
        out = pd.DataFrame({"q_id": q_ids[q_pos[qi]], "s1_id": s_ids[sj]})
        for ix in INDEXES:
            Xs, Xq = mats[ix]
            out[f"cos_{ix}"] = pair_cosine(Xq, Xs, qi, sj)
            kk, rr = rank_of[ix]
            kmax = max(k[ix], cc.k_name_noaddr) if ix == "name" else k[ix]
            if len(kk):
                order = np.argsort(kk)
                pos = np.minimum(np.searchsorted(kk[order], keys), len(kk) - 1)
                hit = kk[order][pos] == keys
                out[f"rank_{ix}"] = np.where(hit, rr[order][pos], kmax).astype(np.int16)
            else:
                out[f"rank_{ix}"] = np.int16(kmax)
        del mats, qb
        yield out


# ------------------------------------------------------------------ report


class RecallReport:
    """Accumulates recall statistics part by part (parts partition the queries)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.true_parts: list[pd.DataFrame] = []
        self.true: pd.DataFrame | None = None
        self.side: pd.Series | None = None
        self.n_queries = 0
        self.found: list[pd.DataFrame] = []
        self.per_query: list[np.ndarray] = []
        self.n_pairs = 0
        self.val_q = self.val_q_leak = self.val_c = self.val_c_leak = 0

    def set_country(self, recs: pd.DataFrame) -> None:
        """Records of the next country (true pairs never cross countries, so the
        report is built one country at a time)."""
        q = recs[recs.source != 1]
        s1_ids = set(recs.entity_id[recs.source == 1])
        self.true = q[(q.owner != "") & q.owner.isin(s1_ids)][
            ["entity_id", "owner", "country", "source", "name_indic", "side"]]
        self.true_parts.append(self.true)
        self.side = recs.set_index("entity_id").side
        self.n_queries += len(q)

    def add(self, part: pd.DataFrame) -> None:
        p = part[["q_id", "s1_id", "rank_name", "rank_addr", "rank_combo"]].astype(
            {"q_id": str, "s1_id": str})
        self.found.append(p.merge(self.true, left_on=["q_id", "s1_id"],
                                  right_on=["entity_id", "owner"]))
        self.per_query.append(p.groupby("q_id").size().to_numpy())
        self.n_pairs += len(p)
        qs = self.side.reindex(p.q_id).to_numpy()
        ss = self.side.reindex(p.s1_id).to_numpy()
        v = qs == "val"
        if v.any():
            leak = pd.Series(ss[v] == "train").groupby(p.q_id.to_numpy()[v]).any()
            self.val_q += len(leak)
            self.val_q_leak += int(leak.sum())
            self.val_c += int(v.sum())
            self.val_c_leak += int((ss[v] == "train").sum())

    def write(self, st: StageRun) -> None:
        cc = self.cfg.candidates
        k = {"name": cc.k_name, "addr": cc.k_addr, "combo": cc.k_combo}
        self.true = pd.concat(self.true_parts, ignore_index=True)
        f = pd.concat(self.found, ignore_index=True) if self.found else self.true.iloc[:0]
        hit = self.true.merge(f[["entity_id", "rank_name", "rank_addr", "rank_combo"]],
                              on="entity_id", how="left")
        hit["found"] = hit.rank_combo.notna()
        by = hit.groupby(["country", "source", "name_indic", "side"]).found.agg(["mean", "size"])
        st.metric("pair_recall", round(float(hit.found.mean()), 5))
        st.metric("pair_recall_by_segment", {" ".join(map(str, i)): [round(float(r["mean"]), 5),
                                                                     int(r["size"])]
                                             for i, r in by.iterrows()})
        st.metric("cluster_recall", round(float(hit.groupby("owner").found.all().mean()), 5))
        at = {}
        for kk in sorted({1, 3, 5, 10, *k.values()}):
            row = {ix: round(float((hit[f"rank_{ix}"] < kk).mean()), 5)
                   for ix in INDEXES if kk <= k[ix]}
            if all(kk <= v for v in k.values()):
                row["union"] = round(float(((hit.rank_name < kk) | (hit.rank_addr < kk)
                                            | (hit.rank_combo < kk)).mean()), 5)
            at[kk] = row
        st.metric("recall_at_k", at)
        pq = np.concatenate(self.per_query) if self.per_query else np.array([0])
        zero = self.n_queries - len(pq)
        pq = np.r_[pq, np.zeros(zero, dtype=pq.dtype)]
        st.metric("candidates_per_query", {"mean": round(float(pq.mean()), 2),
                                           "p50": int(np.median(pq)),
                                           "p95": int(np.quantile(pq, 0.95)),
                                           "max": int(pq.max()), "zero": int(zero)})
        st.metric("candidate_pairs", self.n_pairs)
        if self.val_q:
            # descriptive only: val labels never reach training, so this is not label leakage
            st.metric("val_candidates_crossing_split", {
                "val_queries_with_train_candidate": round(self.val_q_leak / self.val_q, 4),
                "val_candidates_on_train_side": round(self.val_c_leak / max(self.val_c, 1), 4)})


# ------------------------------------------------------------------- stage


def run_candidates(cfg: Config) -> None:
    """One country at a time (records are read with a parquet filter), so only
    that country's records and TF-IDF matrices are in memory."""
    with StageRun(cfg, "candidates") as st:
        splits = (["train"] if cfg.run_train else []) + (["test"] if cfg.run_test else [])
        neighbors: dict[str, dict[str, list[str]]] = {}
        if not cfg.run_train:  # test-only pass: neighbourhoods learned by the train pass
            with open(st.path("region_neighbors.json"), encoding="utf-8") as f:
                neighbors = json.load(f)
        for split in splits:
            cols = RECORD_COLS + (TRAIN_COLS if split == "train" else [])
            path = st.input_path("normalize", f"records_{split}")
            countries = sorted(pd.read_parquet(path, columns=["country"]).country.unique())
            report = RecallReport(cfg) if split == "train" else None
            out_dir = st.path(f"candidates_{split}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*.parquet"):
                old.unlink()
            n_parts = n_rows = 0
            parts_by_country: dict[str, list[str]] = {}
            for country in countries:
                c = pd.read_parquet(path, columns=cols, filters=[("country", "==", country)])
                if split == "train":  # learned on train-side pairs, reused for test
                    side_train = c[(c.source == 1) | (c.side == "train")]
                    neighbors.update(region_neighbors(side_train, cfg.candidates.region_min_share))
                    del side_train
                    report.set_country(c)
                s1, q = c[c.source == 1], c[c.source != 1]
                del c
                with timed(st.log, f"{split} {country}: {len(q):,} queries x {len(s1):,} S1"):
                    for part in generate_country(s1, q, cfg, st.log, neighbors.get(country)):
                        name = f"part-{n_parts:05d}.parquet"
                        part.to_parquet(out_dir / name, index=False)
                        parts_by_country.setdefault(country, []).append(name)
                        n_parts += 1
                        n_rows += len(part)
                        if report is not None:
                            report.add(part)
                del s1, q
            st.register(f"candidates_{split}", out_dir, rows=n_rows, parts=n_parts)
            st.write_json(parts_by_country, f"parts_{split}")
            if split == "train":
                st.write_json(neighbors, "region_neighbors")
                st.metric("region_neighbors_merged", {c: {r: v for r, v in nb.items() if len(v) > 1}
                                                      for c, nb in neighbors.items()})
                with timed(st.log, "recall report"):
                    report.write(st)
