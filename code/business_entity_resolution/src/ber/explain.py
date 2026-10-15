"""``ber explain <entity_id>`` (task T4.3): trace one entity through every stage.

For an S2/S3 record: its raw and normalised fields, split info, every
candidate with retrieval scores, the scorer probability and key features, and
the final decision. For an S1: its fields, the true matches (train), every
query that listed it as a candidate, and what was decided. Missing stages are
reported and skipped, so it works on partial runs.
"""

from __future__ import annotations

import pandas as pd

from .artifacts import StageError, load_manifest
from .config import Config

KEY_FEATURES = ["n_core_eq", "n_ratio", "n_tsr", "n_sub", "n_lev", "l_eq", "l_conflict",
                "a_tsr", "a_bag_eq", "r_eq", "hn_rel", "num_common", "num_q_only",
                "cos_combo_margin", "q_n_cands"]
SHOW_RECORD = ["entity_id", "source", "country", "side", "owner", "name_raw", "name_core",
               "name_legal", "name_alt_core", "name_indic", "addr_raw", "addr_norm", "region",
               "addr_numbers"]


def _stage_file(cfg: Config, stage: str, name: str):
    try:
        m = load_manifest(cfg, stage)
    except StageError as e:
        print(f"  [{stage}] unavailable: {e}")
        return None
    if name not in m["outputs"]:
        return None
    return cfg.run_dir / stage / m["outputs"][name]["file"]


def _read(path, filters=None, columns=None) -> pd.DataFrame:
    """Read a parquet file or a directory of parts. Parts are read one by one:
    their categorical id columns have per-part dictionaries (int16 vs int32
    indices), which pyarrow cannot filter across as one dataset."""
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    parts = [pd.read_parquet(f, filters=filters, columns=columns) for f in files]
    df = pd.concat([p.astype({c: str for c in ("q_id", "s1_id") if c in p}) for p in parts],
                   ignore_index=True) if parts else pd.DataFrame()
    for c in ("q_id", "s1_id"):
        if c in df:
            df[c] = df[c].astype(str)
    return df


def _records(cfg: Config, split: str, ids: list[str]) -> pd.DataFrame:
    p = _stage_file(cfg, "normalize", f"records_{split}")
    if p is None:
        return pd.DataFrame()
    return _read(p, filters=[("entity_id", "in", ids)])


def explain(cfg: Config, entity_id: str) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    split = next((s for s in ("train", "test") if not _records(cfg, s, [entity_id]).empty), None)
    if split is None:
        print(f"{entity_id}: not found in run '{cfg.run_id}' (normalize outputs)")
        return
    rec = _records(cfg, split, [entity_id])
    print(f"=== {entity_id} ({split}, run {cfg.run_id})")
    print(rec[[c for c in SHOW_RECORD if c in rec]].T.to_string(header=False))
    is_s1 = entity_id.startswith("S1-")
    key = "s1_id" if is_s1 else "q_id"

    cp = _stage_file(cfg, "candidates", f"candidates_{split}")
    cands = _read(cp, filters=[(key, "==", entity_id)]) if cp else pd.DataFrame()
    print(f"\n--- candidates ({len(cands)})")
    if not cands.empty:
        other = "q_id" if is_s1 else "s1_id"
        names = _records(cfg, split, cands[other].tolist()).set_index("entity_id")
        cands = cands.join(names[["name_raw", "addr_raw"]], on=other)
        fp = _stage_file(cfg, "features", f"features_{split}")
        if fp:
            f = _read(fp, filters=[(key, "==", entity_id)],
                      columns=["q_id", "s1_id"] + KEY_FEATURES)
            cands = cands.merge(f, on=["q_id", "s1_id"], how="left")
        sname = "scores_val" if split == "train" else "scores_test"
        sp = _stage_file(cfg, "scorer", sname)
        if sp:
            s = _read(sp, filters=[(key, "==", entity_id)], columns=["q_id", "s1_id", "p"])
            cands = cands.merge(s, on=["q_id", "s1_id"], how="left")
        if "owner" in rec and not is_s1:
            cands["is_true"] = cands.s1_id == rec.owner.iloc[0]
        sort = "p" if "p" in cands else "cos_combo"
        print(cands.sort_values(sort, ascending=False).head(15).to_string(index=False))

    if split == "train" and is_s1:
        truth = _read(_stage_file(cfg, "normalize", "records_train"),
                      filters=[("owner", "==", entity_id)], columns=["entity_id", "name_raw",
                                                                    "addr_raw"])
        print(f"\n--- true matches ({len(truth)})")
        print(truth.to_string(index=False))

    dname = "decisions_val" if split == "train" else "decisions_test"
    dp = _stage_file(cfg, "decide", dname)
    if dp:
        d = _read(dp, filters=[(key, "==", entity_id)])
        print(f"\n--- decision: {'matched' if len(d) else 'no match kept'}")
        if len(d):
            print(d.to_string(index=False))
