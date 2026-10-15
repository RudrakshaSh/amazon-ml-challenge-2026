"""Geographic train/val split and the dev slice (task T0.3).

Whole regions (US states / Indian states) are held out, so lookalike
distractors, which sit at nearby addresses, stay on the same side as the
entities they imitate. Within each country:

1. regions are shuffled (seeded), and whole regions are added to *val* until
   ~``val_frac`` of that country's S1 entities are in val;
2. the *dev* slice is a few small whole regions: ~``dev_val_frac`` of the
   country (chosen first, and part of val) and ~``dev_train_frac`` from train
   regions, so a dev run can train and evaluate end-to-end in minutes.

S2/S3 records ("queries") follow their true owner when matched. Unmatched ones
follow the region detected in their own address. Records with no detectable
region are assigned by a stable hash, in the same proportions. Once candidates
exist (T2.2), the real leakage rate (val queries with train-side candidates) is
measured there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..io import read_ground_truth, read_source
from ..log import timed
from ..normalize.geo import UNKNOWN, detect_region

SLICES = (None, "dev")


def stable_unit(ids: pd.Series, seed: int) -> np.ndarray:
    """Deterministic pseudo-uniform [0, 1) value per id (independent of row order)."""
    h = pd.util.hash_pandas_object(ids, index=False, hash_key=f"ber-split-{seed:06d}"[:16])
    return h.to_numpy(dtype=np.uint64) / np.float64(2**64)


def choose_regions(sizes: pd.Series, target: float, rng: np.random.Generator) -> list[str]:
    """Greedily add shuffled regions until their total size reaches ~target.
    A region is skipped if it would overshoot the target by more than 15%.
    If that leaves the pick below half the target (every remaining region is
    too big, e.g. India's smallest real state is ~1.5% of the country), the one
    region closest in size to the shortfall is added."""
    chosen, total = [], 0
    for region in rng.permutation(sorted(sizes.index)):
        if total >= 0.95 * target:
            break
        if total + sizes[region] <= 1.15 * target:
            chosen.append(region)
            total += sizes[region]
    rest = sizes[~sizes.index.isin(chosen)]
    if target > 0 and total < 0.5 * target and len(rest):
        chosen.append((rest - (target - total)).abs().idxmin())
    return chosen


def assign_regions(s1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (country, region) with side (train/val) and dev flag."""
    sc = cfg.split
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for country in sorted(s1.country.unique()):
        c = s1[s1.country == country]
        sizes = c[c.geo != UNKNOWN].groupby("geo").size()
        n = len(c)
        # dev-val regions first (small ones), so the dev slice always has val data;
        # they count towards val, which is then filled up to val_frac.
        dev_val = set(choose_regions(sizes, sc.dev_val_frac * n, rng))
        rest = sizes[~sizes.index.isin(dev_val)]
        val = dev_val | set(choose_regions(rest, sc.val_frac * n - sizes[list(dev_val)].sum(),
                                           rng))
        train_sizes = sizes[~sizes.index.isin(val)]
        dev = dev_val | set(choose_regions(train_sizes, sc.dev_train_frac * n, rng))
        for region, size in sizes.items():
            rows.append({"country": country, "geo": region, "n_s1": int(size),
                         "side": "val" if region in val else "train", "dev": region in dev})
    return pd.DataFrame(rows)


def hold_regions(regions: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Whole train-side regions set aside for stage 2 (v2.4, ``rerank.hold_frac``):
    per country, shuffled regions are added until ~``frac`` of the country's S1 is
    covered (same greedy rule as the val split, own seed). Stage 1 then trains
    without them, so their stage-1 scores are out-of-sample like val's and test's,
    and stage 2 learns from val + these regions (more regions, less region-specific
    patterns). Returns rows (country, geo, n_s1) of ``regions``."""
    rng = np.random.default_rng(seed + 11)
    out = []
    for country in sorted(regions.country.unique()):
        r = regions[regions.country == country]
        sizes = r[r.side == "train"].set_index("geo").n_s1
        chosen = choose_regions(sizes, frac * r.n_s1.sum(), rng) if frac > 0 else []
        out.append(r[r.geo.isin(chosen)][["country", "geo", "n_s1"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["country", "geo", "n_s1"])


def hold_queries(queries: pd.DataFrame, s1: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    """Train-side queries (S2/S3 records) of the hold regions, as rows (entity_id,
    country, geo = the hold region they belong to).

    The val split's rule (v2.5): a matched query belongs to its owner's region, an
    unmatched one to the region in its own address. Every hold S1 then has all its
    true copies loaded, also those whose address names no region (4-5 % of true
    copies in US/India: the records that search the whole country, like 35 % of
    France's) or another region; the query's own region alone (v2.4) missed 5 % of
    the hold S1's copies and left their groups incomplete.
    queries: entity_id, country, geo, side, owner ("" if unmatched); s1: entity_id, geo."""
    tr = queries.loc[(queries.side == "train").to_numpy(), ["entity_id", "country", "geo", "owner"]]
    owner_geo = s1.set_index("entity_id").geo.reindex(tr.owner.to_numpy()).to_numpy(dtype=object)
    matched = (tr.owner != "").to_numpy()
    geo = np.where(matched, owner_geo, tr.geo.to_numpy(dtype=object))
    out = pd.DataFrame({"entity_id": tr.entity_id.to_numpy(), "country": tr.country.to_numpy(),
                        "geo": geo})
    return out.merge(hold[["country", "geo"]].drop_duplicates(), on=["country", "geo"])


def _hash_fill(df: pd.DataFrame, mask: pd.Series, regions: pd.DataFrame, cfg: Config) -> None:
    """Side/dev by stable hash for rows in ``mask`` (no usable region), in the
    proportions each country actually got from the region split."""
    if not mask.any():
        return
    u = pd.Series(stable_unit(df.entity_id, cfg.seed), index=df.index)
    for country, g in regions.groupby("country"):
        m = mask & (df.country == country)
        tot = g.n_s1.sum()
        pv = g.n_s1[g.side == "val"].sum() / tot
        qv = g.n_s1[(g.side == "val") & g.dev].sum() / max(g.n_s1[g.side == "val"].sum(), 1)
        qt = g.n_s1[(g.side == "train") & g.dev].sum() / max(g.n_s1[g.side == "train"].sum(), 1)
        um = u[m]
        df.loc[m, "side"] = np.where(um < pv, "val", "train")
        df.loc[m, "dev"] = (um < pv * qv) | ((um >= pv) & (um < pv + (1 - pv) * qt))
    df.loc[mask, "basis"] = "hash"


def make_split(s1: pd.DataFrame, queries: pd.DataFrame, cfg: Config):
    """s1: entity_id, country, geo.  queries: entity_id, country, geo, owner ("" if
    unmatched). Returns (s1_split, query_split, regions)."""
    regions = assign_regions(s1, cfg)
    lut = regions.set_index(["country", "geo"])[["side", "dev"]]

    s1 = s1.copy()
    s1 = s1.join(lut, on=["country", "geo"])
    s1["basis"] = "region"
    _hash_fill(s1, s1.side.isna(), regions, cfg)
    s1["dev"] = s1.dev.astype(bool)

    q = queries.copy()
    owner = s1.set_index("entity_id")[["side", "dev"]]
    q = q.join(owner, on="owner")
    q["basis"] = np.where(q.side.notna(), "owner", "")  # works on empty frames too
    own = q.side.isna()
    q.loc[own, ["side", "dev"]] = q.loc[own, ["country", "geo"]].join(
        lut, on=["country", "geo"])[["side", "dev"]].to_numpy()
    q.loc[own & q.side.notna(), "basis"] = "own_region"
    _hash_fill(q, q.side.isna(), regions, cfg)
    q["dev"] = q.dev.astype(bool)
    return s1, q, regions


def slice_mask(df: pd.DataFrame, cfg: Config, side: str | None = None) -> pd.Series:
    """Rows of a split frame that belong to the configured slice (and side)."""
    if cfg.data.slice not in SLICES:
        raise ValueError(f"unknown data.slice {cfg.data.slice!r}; valid: {SLICES}")
    m = pd.Series(True, index=df.index)
    if cfg.data.slice == "dev":
        m &= df.dev
    if side is not None:
        m &= df.side == side
    return m


# ------------------------------------------------------------------ stage


def _report(st: StageRun, s1: pd.DataFrame, q: pd.DataFrame, regions: pd.DataFrame) -> None:
    for country, g in s1.groupby("country"):
        r = regions[regions.country == country]
        qc = q[q.country == country]
        matched = qc[qc.owner != ""]
        known = matched[matched.geo != UNKNOWN]
        owner_geo = s1.set_index("entity_id").geo
        st.metric(f"{country}.regions", {"total": len(r), "val": int((r.side == "val").sum()),
                                         "dev": int(r.dev.sum())})
        st.metric(f"{country}.val_regions", sorted(r.geo[r.side == "val"]))
        st.metric(f"{country}.dev_regions", sorted(r.geo[r.dev]))
        st.metric(f"{country}.s1", {
            "n": len(g), "val_frac": round(float((g.side == "val").mean()), 4),
            "dev_train": int((g.dev & (g.side == "train")).sum()),
            "dev_val": int((g.dev & (g.side == "val")).sum()),
            "unknown_region_frac": round(float((g.geo == UNKNOWN).mean()), 5)})
        for src in ("S2", "S3"):
            s = qc[qc.entity_id.str.startswith(src)]
            un = s[s.owner == ""]
            ks = known[known.entity_id.str.startswith(src)]
            agree = (ks.geo.to_numpy() == owner_geo.loc[ks.owner].to_numpy()).mean()
            st.metric(f"{country}.{src}", {
                "n": len(s), "val_frac": round(float((s.side == "val").mean()), 4),
                "dev": int(s.dev.sum()),
                "unmatched": len(un),
                "unmatched_basis": un.basis.value_counts().to_dict(),
                # how often a record's own detected region equals its owner's region;
                # proxy for how well unmatched records are placed
                "matched_own_region_agrees_with_owner": round(float(agree), 4),
                "matched_region_unknown_frac": round(float((matched.entity_id.str.startswith(src)
                                                            & (matched.geo == UNKNOWN)).sum()
                                                           / max(len(s) - len(un), 1)), 4)})


def run_split(cfg: Config) -> None:
    with StageRun(cfg, "split") as st:
        if cfg.data.slice is not None:
            st.log.info("data.slice=%r ignored: the split stage always uses the full train "
                        "set (it defines the slices)", cfg.data.slice)
        cols = ["entity_id", "business_address", "country"]
        with timed(st.log, "read train sources + ground truth"):
            s1 = read_source(cfg, "train", 1)[cols]
            q = pd.concat([read_source(cfg, "train", n)[cols] for n in (2, 3)],
                          ignore_index=True)
            gt = read_ground_truth(cfg)
        with timed(st.log, "owner map"):
            m = gt[gt.matched_entity_ids != ""]
            pairs = m.assign(other=m.matched_entity_ids.str.split(",")).explode("other")
            owner = pd.Series(pairs.source1_entity_id.to_numpy(), index=pairs.other.to_numpy())
            if not owner.index.is_unique:
                raise ValueError("ground truth assigns an S2/S3 record to several S1s")
            q["owner"] = q.entity_id.map(owner).fillna("")
        with timed(st.log, f"detect regions for {len(s1) + len(q):,} addresses"):
            s1["geo"] = [detect_region(a, c) for a, c in zip(s1.business_address, s1.country)]
            q["geo"] = [detect_region(a, c) for a, c in zip(q.business_address, q.country)]
        s1 = s1.drop(columns="business_address")
        q = q.drop(columns="business_address")

        with timed(st.log, "make split"):
            s1s, qs, regions = make_split(s1, q, cfg)
        _report(st, s1s, qs, regions)
        st.write_parquet(s1s, "s1")
        st.write_parquet(qs, "queries")
        st.write_parquet(regions, "regions")
