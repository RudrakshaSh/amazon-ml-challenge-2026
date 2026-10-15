"""``ber features`` (task T3.3): features for every candidate pair.

Train split: all val-side queries plus a ``train_query_frac`` sample of
train-side queries (the model needs far fewer pairs than exist), labelled
``y = (s1_id == owner)``. Test split: every query.

Outputs ``features_<split>/`` (parquet parts) with q_id, s1_id, [y, side] and
the feature columns of :mod:`ber.features.build`.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from ..artifacts import StageError, StageRun, load_manifest
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed
from .build import KeyIndex, address_key, build_features, initials

REC_COLS = ["entity_id", "source", "country", "name_core", "name_compact", "name_legal",
            "name_pre_core", "name_alt_core", "name_is_domain", "name_has_alias", "name_indic",
            "name_n_indic_oov", "addr_norm", "addr_bag", "region", "addr_numbers",
            "addr_has_box"]


def learn_fillers(frames, min_rate: float, min_ratio: float) -> list[str]:
    """Filler words the noise appends to S2/S3 names ("Riverside", "Services",
    "Participations", "Smt" ...), learned without labels: tokens whose share of
    S2/S3 core names (per country and split) is >= ``min_ratio`` times their share
    of S1 core names, and >= ``min_rate``. A copy of an S1 name keeps its tokens at
    the S1 rate, so only tokens the noise injects stand out. The union over every
    (split, country) is used everywhere, so train and test features agree."""
    out: set[str] = set()
    for df in frames:
        for _, g in df.groupby("country"):
            s1, q = g.name_core[g.source == 1], g.name_core[g.source != 1]
            if len(s1) == 0 or len(q) == 0:
                continue
            cs = Counter(t for n in s1 for t in set(n.split()))
            cq = Counter(t for n in q for t in set(n.split()))
            for t, c in cq.items():
                rate = c / len(q)
                if rate >= min_rate and rate >= min_ratio * (cs[t] + 1) / len(s1):
                    out.add(t)
    return sorted(out)


def learn_brand_units(frames) -> list[str]:
    """Syllables of the made-up brand names the noise substitutes for a whole
    business name ("Grand Physical Therapy" -> "Jaxdrexquo", "Eye Peak Group" ->
    "Jaxvantage"; ~1.6% of true pairs). Learned from train true pairs whose query
    name is one token unknown to the S1 vocabulary and unlike the owner's name:
    the maximal prefixes shared by many of them (tavo, jax, kelo, zeph, ...).
    Such names can only be matched by address; the model needs to know that."""
    from rapidfuzz import fuzz, process
    pre: Counter = Counter()
    for recs in frames:  # train records, one country at a time
        s1 = recs[recs.source == 1]
        vocab = set(t for n in s1.name_core for t in n.split())
        q = recs[(recs.source != 1) & (recs.owner != "") & ~recs.name_indic]
        q = q[~q.name_core.str.contains(" ", regex=False) & (q.name_core.str.len() >= 6)
              & ~q.name_core.isin(vocab) & q.name_compact.str.fullmatch("[a-z]+")]
        s_compact = s1.set_index("entity_id").name_compact.reindex(q.owner).fillna("")
        sim = process.cpdist(q.name_compact.tolist(), s_compact.tolist(), scorer=fuzz.ratio,
                             workers=-1)
        words = q.name_compact[sim < 40]
        pre.update(w[:n] for w in words for n in range(2, min(len(w), 8) + 1))
    if not pre:
        return []
    thr = 0.25 * float(np.median([c for _, c in pre.most_common(40)]))
    letters = "abcdefghijklmnopqrstuvwxyz"
    return sorted(p for p, c in pre.items()
                  if c >= thr and all(pre.get(p + ch, 0) < 0.9 * c for ch in letters))


def is_brand(compact: pd.Series, units: frozenset[str], max_len: int = 8) -> pd.Series:
    """Whether each name (compact form) is a concatenation of brand syllables."""
    def seg(w: str) -> bool:
        if len(w) < 5 or not w.isalpha():
            return False
        ok = [True] + [False] * len(w)
        for i in range(1, len(w) + 1):
            ok[i] = any(ok[j] and w[j:i] in units for j in range(max(0, i - max_len), i))
        return ok[-1]
    return compact.map(seg).astype(np.int8)


def name_key(core: pd.Series, fillers: frozenset[str]) -> pd.Series:
    """Sorted unique core tokens without filler words (order- and filler-free
    form of the name); the plain sorted tokens if nothing else is left."""
    def key(n: str) -> str:
        toks = sorted(set(n.split()))
        kept = [t for t in toks if t not in fillers]
        return " ".join(kept or toks)
    return core.map(key)


def token_idf(keys: pd.Series) -> dict[str, float]:
    """Smoothed IDF of name-key tokens over S1 records."""
    df = Counter(t for n in keys for t in n.split())
    n = len(keys)
    return {t: float(np.log((n + 1) / (c + 1)) + 1.0) for t, c in df.items()}


def record_extras(recs: pd.DataFrame, fillers: list[str] | None = None,
                  brand_units: list[str] | None = None) -> pd.DataFrame:
    """Per-record extras: initials, share of name tokens unknown to the S1 name
    vocabulary of the country (gibberish brand names score high), for S1 how
    many S1 of the country share the exact core name (ambiguity), and (v2) the
    filler-free name key and the made-up-brand flag."""
    recs = recs.copy()
    recs["initials"] = initials(recs.name_core)
    if fillers is not None:
        recs["name_key"] = name_key(recs.name_core, frozenset(fillers))
    if brand_units is not None:
        recs["brand"] = is_brand(recs.name_compact, frozenset(brand_units))
    s1 = recs[recs.source == 1]
    vocab = {c: Counter(t for n in g.name_core for t in set(n.split()))
             for c, g in s1.groupby("country")}
    recs["vocab_oov"] = [
        (sum(vocab.get(c, {}).get(t, 0) == 0 for t in n.split()) / len(n.split())) if n else 1.0
        for n, c in zip(recs.name_core, recs.country)]
    freq = s1.groupby(["country", "name_core"]).size().rename("name_freq")
    recs = recs.join(freq, on=["country", "name_core"])
    recs["name_freq"] = recs.name_freq.fillna(0).astype(np.int32)
    return recs


def s1_context(cand_dir) -> pd.DataFrame:
    """Per S1: number of queries listing it (s_n_cands) and number of queries
    whose best combo candidate it is (s_n_top1), over the full candidate set."""
    n_c, n_t = Counter(), Counter()
    for f in sorted(cand_dir.glob("*.parquet")):
        p = pd.read_parquet(f, columns=["q_id", "s1_id", "cos_combo"])
        p["s1_id"] = p.s1_id.astype(str)
        n_c.update(p.s1_id.value_counts().to_dict())
        best = p.loc[p.groupby("q_id", observed=True).cos_combo.idxmax(), "s1_id"]
        n_t.update(best.value_counts().to_dict())
    idx = list(n_c)
    return pd.DataFrame({"s_n_cands": [n_c[i] for i in idx],
                         "s_n_top1": [n_t.get(i, 0) for i in idx]}, index=idx)


def _query_chunks(part: pd.DataFrame, max_rows: int):
    """Split a candidate part (rows grouped by query) at query boundaries."""
    q = part.q_id.to_numpy()
    cuts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    start = 0
    for i in range(1, len(cuts)):
        if cuts[i] - start >= max_rows or i == len(cuts) - 1:
            yield part.iloc[start:cuts[i]]
            start = cuts[i]


def join_records(chunk: pd.DataFrame, rec: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Attach q_/s_ record fields and per-S1 context to candidate pairs."""
    qf = rec.reindex(chunk.q_id.to_numpy()).add_prefix("q_").reset_index(drop=True)
    sf = rec.reindex(chunk.s1_id.to_numpy()).add_prefix("s_").reset_index(drop=True)
    c = ctx.reindex(chunk.s1_id.to_numpy()).fillna(0).reset_index(drop=True)
    return pd.concat([chunk.reset_index(drop=True), qf, sf, c], axis=1)


def _has_output(st: StageRun, stage: str, name: str) -> bool:
    try:
        return name in load_manifest(st.cfg, stage)["outputs"]
    except StageError:
        return False


def _by_country(path, columns: list[str]):
    """(country, records) one country at a time (parquet filter: bounded memory)."""
    for c in sorted(pd.read_parquet(path, columns=["country"]).country.unique()):
        yield c, pd.read_parquet(path, columns=columns, filters=[("country", "==", c)])


def _learn_vocabularies(st: StageRun, cfg: Config) -> tuple[list[str], list[str]]:
    """Filler words (both splits whenever test records exist, also in a train-only
    pass, so the train and test passes agree) and brand syllables (train)."""
    fc = cfg.features
    with timed(st.log, "learn filler words"):
        paths = [st.input_path("normalize", f"records_{sp}") for sp in ("train", "test")
                 if _has_output(st, "normalize", f"records_{sp}")]
        fillers = learn_fillers((r for p in paths for _, r in _by_country(
            p, ["source", "country", "name_core"])), fc.filler_min_rate, fc.filler_min_ratio)
    st.write_json(fillers, "fillers")
    st.metric("fillers", " ".join(fillers))
    with timed(st.log, "learn brand syllables"):
        cols = ["entity_id", "source", "owner", "name_core", "name_compact", "name_indic"]
        brand_units = learn_brand_units(r for _, r in _by_country(
            st.input_path("normalize", "records_train"), cols))
    st.write_json(brand_units, "brand_units")
    st.metric("brand_units", " ".join(brand_units))
    return fillers, brand_units


def _country_records(recs: pd.DataFrame, fillers, brand_units):
    """Extras + v2 key indexes for one country's records."""
    recs = record_extras(recs, fillers, brand_units)
    kidx = aidx = None
    if fillers is not None:
        s1m = (recs.source == 1).to_numpy()
        kidx = KeyIndex(recs.name_key, token_idf(recs.name_key[s1m]))
        akey = address_key(recs.addr_bag)
        aidx = KeyIndex(akey, token_idf(akey[s1m]))
        recs["kpos"] = np.arange(len(recs), dtype=np.int32)
    return recs, kidx, aidx


def _read_country_records(path, country: str, columns: list[str],
                          query_ids: set[str] | None = None) -> pd.DataFrame:
    """One country's S1 records plus (if given) only the queries in ``query_ids``,
    read with Arrow filters (the other queries never reach memory)."""
    import pyarrow.dataset as ds
    dset = ds.dataset(path, format="parquet")
    f_c = ds.field("country") == country
    s1 = dset.to_table(columns=columns, filter=f_c & (ds.field("source") == 1))
    if query_ids is None:
        return s1.to_pandas()
    q = dset.to_table(columns=columns, filter=f_c & (ds.field("source") != 1)
                      & ds.field("entity_id").isin(sorted(query_ids)))
    import pyarrow as pa
    return pa.concat_tables([s1, q]).to_pandas()


def run_features_hold(cfg: Config) -> None:
    """``ber features-hold`` (v2.4): features for every candidate pair of the
    queries in the held-out train-side regions (``rerank.hold_frac``, see
    :func:`ber.eval.split.hold_regions`), which stage 2 trains on next to val.

    Identical to what ``ber features`` computes for these queries: the filler words
    and brand syllables are read from its outputs, and every corpus statistic
    (token IDF, name frequency, name vocabulary, per-S1 candidate counts) comes from
    all S1 of the country and the full candidate set. Only the records needed are
    loaded: every S1 of the country and the hold queries (about a third of the
    memory of the full stage). Output ``features_hold/`` has side = "hold"."""
    from ..eval.split import hold_queries, hold_regions
    fc = cfg.features
    with StageRun(cfg, "features_hold") as st:
        regions = st.read_input("split", "regions", run=cfg.data.split_run)
        hold = hold_regions(regions, cfg.rerank.hold_frac, cfg.seed)
        if hold.empty:
            raise ValueError("rerank.hold_frac selects no region: set it > 0")
        st.write_parquet(hold, "hold_regions")
        st.metric("hold_regions", {c: sorted(g.geo) for c, g in hold.groupby("country")})
        qs = st.read_input("split", "queries", run=cfg.data.split_run,
                           columns=["entity_id", "country", "geo", "side", "owner"])
        s1s = st.read_input("split", "s1", run=cfg.data.split_run, columns=["entity_id", "geo"])
        hq = hold_queries(qs, s1s, hold)
        st.write_parquet(hq, "hold_queries")
        del qs, s1s
        fillers = brand_units = None
        if fc.name_key:
            with open(st.input_path("features", "fillers"), encoding="utf-8") as fh:
                fillers = json.load(fh)
            with open(st.input_path("features", "brand_units"), encoding="utf-8") as fh:
                brand_units = json.load(fh)
        rec_path = st.input_path("normalize", "records_train")
        cand_dir = st.input_path("candidates", "candidates_train")
        with timed(st.log, "train: per-S1 candidate context"):
            ctx = s1_context(cand_dir)
        with open(st.input_path("candidates", "parts_train"), encoding="utf-8") as fh:
            parts_by_country = json.load(fh)
        out_dir = st.path("features_hold")
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.parquet"):
            old.unlink()
        n_parts = n_rows = n_pos = 0
        for country, g in hq.groupby("country"):
            ids = set(g.entity_id)
            with timed(st.log, f"hold {country}: records + extras ({len(ids)} queries)"):
                recs = _read_country_records(rec_path, country, REC_COLS + ["owner"], ids)
                recs, kidx, aidx = _country_records(recs, fillers, brand_units)
                rec = recs.set_index("entity_id")
                del recs
            with timed(st.log, f"hold {country}: features"):
                for name in parts_by_country.get(country, []):
                    part = pd.read_parquet(cand_dir / name).astype({"q_id": str, "s1_id": str})
                    part = part[part.q_id.isin(ids)]
                    if part.empty:
                        continue
                    for chunk in _query_chunks(part, fc.chunk_rows):
                        d = join_records(chunk, rec, ctx)
                        feats = build_features(d, kidx, aidx)
                        keys = d[["q_id", "s1_id"]].copy()
                        keys["y"] = (d.s1_id == d.q_owner).astype(np.int8)
                        keys["side"] = "hold"
                        n_pos += int(keys.y.sum())
                        out = pd.concat([keys, feats], axis=1)
                        out.to_parquet(out_dir / f"part-{n_parts:05d}.parquet", index=False)
                        n_parts += 1
                        n_rows += len(out)
                        del d, feats, out
            del rec, kidx, aidx
        st.register("features_hold", out_dir, rows=n_rows, parts=n_parts)
        st.metric("hold_pairs", n_rows)
        st.metric("hold_queries", len(hq))
        st.metric("hold_positive_rate", round(n_pos / max(n_rows, 1), 4))


def run_features(cfg: Config) -> None:
    """Candidate parts are grouped by country (``parts_<split>.json`` of the
    candidates stage), so only one country's records are in memory at a time."""
    fc = cfg.features
    with StageRun(cfg, "features") as st:
        splits = (["train"] if cfg.run_train else []) + (["test"] if cfg.run_test else [])
        fillers = brand_units = None
        if fc.name_key:
            fillers, brand_units = _learn_vocabularies(st, cfg)
        for split in splits:
            extra = ["side", "owner"] if split == "train" else []
            rec_path = st.input_path("normalize", f"records_{split}")
            cand_dir = st.input_path("candidates", f"candidates_{split}")
            with timed(st.log, f"{split}: per-S1 candidate context"):
                ctx = s1_context(cand_dir)
            with open(st.input_path("candidates", f"parts_{split}"), encoding="utf-8") as fh:
                parts_by_country = json.load(fh)
            out_dir = st.path(f"features_{split}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*.parquet"):
                old.unlink()
            n_parts = n_rows = n_pos = n_sel_train = n_sel_val = 0
            for country, recs in _by_country(rec_path, REC_COLS + extra):
                with timed(st.log, f"{split} {country}: records + extras"):
                    recs, kidx, aidx = _country_records(recs, fillers, brand_units)
                    if split == "train":
                        q = recs[recs.source != 1]
                        u = stable_unit(q.entity_id, cfg.seed + 1)
                        keep = (q.side == "val").to_numpy() | (u < fc.train_query_frac)
                        selected = set(q.entity_id[keep])
                        n_sel_train += int((keep & (q.side == "train")).sum())
                        n_sel_val += int((q.side == "val").sum())
                        del q
                    rec = recs.set_index("entity_id")
                    del recs
                with timed(st.log, f"{split} {country}: features"):
                    for name in parts_by_country.get(country, []):
                        part = pd.read_parquet(cand_dir / name).astype({"q_id": str, "s1_id": str})
                        if split == "train":
                            part = part[part.q_id.isin(selected)]
                        if part.empty:
                            continue
                        for chunk in _query_chunks(part, fc.chunk_rows):
                            d = join_records(chunk, rec, ctx)
                            feats = build_features(d, kidx, aidx)
                            keys = d[["q_id", "s1_id"]].copy()
                            if split == "train":
                                keys["y"] = (d.s1_id == d.q_owner).astype(np.int8)
                                keys["side"] = d.q_side.to_numpy()
                                n_pos += int(keys.y.sum())
                            out = pd.concat([keys, feats], axis=1)
                            out.to_parquet(out_dir / f"part-{n_parts:05d}.parquet", index=False)
                            n_parts += 1
                            n_rows += len(out)
                            del d, feats, out
                        st.log.debug("features %s: %d rows so far", name, n_rows)
                del rec, kidx, aidx
            if split == "train":
                st.metric("train_queries_selected", {"train": n_sel_train, "val": n_sel_val})
            st.register(f"features_{split}", out_dir, rows=n_rows, parts=n_parts)
            st.metric(f"{split}_pairs", n_rows)
            if split == "train":
                st.metric("train_positive_rate", round(n_pos / max(n_rows, 1), 4))
