"""Cross-encoder data (exploration of a transformer re-scorer next to the GBM cascade).

(1) Training pairs from stage 1's own regions: train-side records outside the hold regions
    and NOT in stage 1's training sample (stable_unit(q_id, seed + 4) >= 0.14), so their p1
    (stage-1 model) is out-of-sample; kept at p1 >= 0.001, exactly how stage 2's inputs
    are chosen. Stage 2 trains on val + hold and is applied to test, so a cross-encoder
    trained here is out-of-sample on every pair stage 2 ever sees.
(2) The stage-2 pairs to score: val and hold (rerank outputs of a run without the
    cross-encoder, in their row order) and test (its scores_test parts, one per country).
Text of a record = normalised name core + legal form " ; " normalised address + region.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ARTIFACTS as A, DATA as OUT, RUN_FEATURES, RUN_RERANK, RUN_SCORER, SRC  # noqa: E402

sys.path.insert(0, str(SRC))
from ber.eval.split import stable_unit  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
SEED, TRAIN_FRAC, MIN_P1 = 42, 0.14, 0.001


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def texts(split: str, ids) -> pd.Series:
    """entity_id -> "name core legal ; address region" for the given ids."""
    d = ds.dataset(A / RUN_FEATURES / "normalize" / f"records_{split}", format="parquet")
    arr = pa.array(sorted(set(ids)), type=pa.large_string())
    t = d.to_table(columns=["entity_id", "name_core", "name_legal", "addr_norm", "region"],
                   filter=ds.field("entity_id").isin(arr))
    df = t.to_pandas()
    name = (df.name_core + " " + df.name_legal).str.strip()
    addr = (df.addr_norm + " " + df.region).str.strip()
    return pd.Series((name + " ; " + addr).to_numpy(), index=df.entity_id.to_numpy())


def with_text(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    tx = texts(split, np.concatenate([pairs.q_id.to_numpy(), pairs.s1_id.to_numpy()]))
    out = pairs.assign(q_text=tx.reindex(pairs.q_id.to_numpy()).to_numpy(),
                       s_text=tx.reindex(pairs.s1_id.to_numpy()).to_numpy())
    if out.q_text.isna().any() or out.s_text.isna().any():
        raise RuntimeError(f"{split}: records missing for some pair ids")
    return out


def train_pairs() -> pd.DataFrame:
    model = lgb.Booster(model_file=str(A / RUN_SCORER / "scorer" / "model.txt"))
    feats = model.feature_name()
    hold = pd.read_parquet(A / RUN_SCORER / "features_hold" / "hold_queries.parquet",
                           columns=["entity_id"]).entity_id.astype(str)
    hold = pa.array(sorted(hold), type=pa.large_string())
    rows, n_seen = [], 0
    files = sorted((A / RUN_FEATURES / "features" / "features_train").glob("*.parquet"))
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=["q_id", "side"])
        m = pc.equal(t.column("side"), "train").to_numpy(zero_copy_only=False)
        if not m.any():
            continue
        q = t.column("q_id").to_pandas().astype(str)
        m &= stable_unit(q, SEED + 4) >= TRAIN_FRAC
        m &= ~pc.is_in(t.column("q_id").cast(pa.large_string()),
                       value_set=hold).to_numpy(zero_copy_only=False)
        if not m.any():
            continue
        d = pd.read_parquet(f, columns=["q_id", "s1_id", "y"] + feats)[m]
        n_seen += len(d)
        p = model.predict(np.ascontiguousarray(d[feats].to_numpy(dtype=np.float32)))
        k = p >= MIN_P1
        rows.append(pd.DataFrame({"q_id": d.q_id.astype(str).to_numpy()[k],
                                  "s1_id": d.s1_id.astype(str).to_numpy()[k],
                                  "y": d.y.to_numpy()[k].astype(np.int8),
                                  "p1": p[k].astype(np.float32)}))
        if i % 20 == 0:
            log(f"  part {i}/{len(files)}: {n_seen:,} pairs scored, {sum(map(len, rows)):,} kept")
        del d, t
    out = pd.concat(rows, ignore_index=True)
    log(f"train pool: {n_seen:,} pairs scored, {len(out):,} kept (p1 >= {MIN_P1}), "
        f"{out.q_id.nunique():,} records, positive rate {out.y.mean():.3f}")
    return out


def main():
    log("training pairs")
    tr = with_text(train_pairs(), "train")
    tr.to_parquet(OUT / "train.parquet", index=False)
    log(f"wrote train.parquet {len(tr):,}")
    del tr
    for side in ("val", "hold"):
        p = pd.read_parquet(A / RUN_RERANK / "rerank" / f"scores_{side}.parquet",
                            columns=["q_id", "s1_id", "y"]).astype({"q_id": str, "s1_id": str})
        with_text(p, "train").to_parquet(OUT / f"{side}.parquet", index=False)
        log(f"wrote {side}.parquet {len(p):,}")
    for f in sorted((A / RUN_RERANK / "rerank" / "scores_test").glob("part-*.parquet")):
        p = pd.read_parquet(f, columns=["q_id", "s1_id"]).astype({"q_id": str, "s1_id": str})
        with_text(p, "test").to_parquet(OUT / f"test_{f.stem.split('-', 1)[1]}.parquet", index=False)
        log(f"wrote test {f.stem} {len(p):,}")


if __name__ == "__main__":
    main()
