"""``ber scorer`` (task T3.4): LightGBM pair scorer.

Trains on train-side candidate pairs, early-stops on val-side pairs, then
scores the val pairs (``scores_val.parquet``) and every test pair
(``scores_test/``). Saves the model, feature list and feature importance.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed

KEY_COLS = ["q_id", "s1_id", "y", "side"]
CATEGORICAL = ["hn_rel", "q_source"]


def feature_columns(df: pd.DataFrame, drop: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Feature columns of a features frame; ``drop`` are feature names or name
    prefixes ending in ``*`` (``scorer.drop_features``, for ablations)."""
    exact = {d for d in drop if not d.endswith("*")}
    prefixes = tuple(d[:-1] for d in drop if d.endswith("*"))
    return [c for c in df.columns if c not in KEY_COLS and c not in exact
            and not (prefixes and c.startswith(prefixes))]


def train_model(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                feats: list[str], cfg: Config, log) -> lgb.Booster:
    """LightGBM on float32 matrices (no pandas copies: LightGBM reads float32
    arrays in place, a mixed-dtype frame would be converted to float64)."""
    sc = cfg.scorer
    params = {
        "objective": "binary", "learning_rate": sc.learning_rate, "num_leaves": sc.num_leaves,
        "min_data_in_leaf": sc.min_data_in_leaf, "feature_fraction": sc.feature_fraction,
        "bagging_fraction": sc.bagging_fraction, "bagging_freq": 1, "lambda_l2": sc.lambda_l2,
        "seed": cfg.seed, "verbose": -1,
        "num_threads": sc.n_threads or max(1, (os.cpu_count() or 4) - 2),
    }
    cat = [c for c in CATEGORICAL if c in feats]
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, feature_name=feats, categorical_feature=cat, reference=dtr)
    dtr.construct()  # binned copy (~1 byte/value); the float32 input can then be freed
    dva.construct()
    del Xtr, Xva
    return lgb.train(params, dtr, num_boost_round=sc.num_boost_round, valid_sets=[dva],
                     valid_names=["val"],
                     callbacks=[lgb.early_stopping(sc.early_stopping, verbose=False),
                                lgb.log_evaluation(100)])


def _matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Feature matrix as one C-contiguous float32 array (categoricals are small
    integers, exact in float32)."""
    return np.ascontiguousarray(df[feats].to_numpy(dtype=np.float32))


def load_sampled(path, cfg: Config, log, feats: list[str],
                 exclude: set[str] | None = None) -> dict:
    """Stream feature parts; keep a query-level sample of train-side rows and of
    val-side rows (the latter for early stopping) as float32 matrices, so memory
    stays bounded (~4 bytes per feature per sampled row). Two passes: the sample
    masks first (ids only), then the rows go straight into preallocated matrices
    (no concatenation copy). Queries in ``exclude`` (the held-out regions of
    ``rerank.hold_frac``) never enter the training sample."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sc = cfg.scorer
    files = sorted(path.glob("*.parquet"))
    # the excluded ids as one Arrow array, built once (a Python set handed to
    # Series.isin is converted again for every part: minutes instead of seconds)
    excl = pa.array(sorted(exclude), type=pa.large_string()) if exclude else None
    masks = []
    for f in files:
        d = pd.read_parquet(f, columns=["q_id", "side"])
        q = d.q_id.astype(str)
        u = stable_unit(q, cfg.seed + 4)
        t = ((d.side == "train").to_numpy() & (u < sc.train_query_frac))
        if excl is not None:
            qa = pq.read_table(f, columns=["q_id"]).column("q_id").cast(pa.large_string())
            t &= ~pc.is_in(qa, value_set=excl).to_numpy(zero_copy_only=False)
        e = ((d.side == "val").to_numpy() & (u < sc.es_query_frac))
        masks.append((t, e))
        del d, q
    n_t = int(sum(t.sum() for t, _ in masks))
    n_e = int(sum(e.sum() for _, e in masks))
    out = {"Xtr": np.empty((n_t, len(feats)), np.float32), "ytr": np.empty(n_t, np.int8),
           "Xes": np.empty((n_e, len(feats)), np.float32), "yes": np.empty(n_e, np.int8)}
    it = ie = 0
    for f, (t, e) in zip(files, masks):
        if not (t.any() or e.any()):
            continue
        d = pd.read_parquet(f)
        if t.any():
            k = int(t.sum())
            out["Xtr"][it:it + k] = d.loc[t, feats].to_numpy(dtype=np.float32)
            out["ytr"][it:it + k] = d.y.to_numpy()[t]
            it += k
        if e.any():
            k = int(e.sum())
            out["Xes"][ie:ie + k] = d.loc[e, feats].to_numpy(dtype=np.float32)
            out["yes"][ie:ie + k] = d.y.to_numpy()[e]
            ie += k
        del d
    log.info("sampled %d train rows, %d early-stopping rows", n_t, n_e)
    return out


def predict_side(model: lgb.Booster, path, side: str, feats: list[str]) -> pd.DataFrame:
    """Scores for every row of one side, streaming the parts."""
    out = []
    for f in sorted(path.glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.side == side]
        if d.empty:
            continue
        out.append(d[["q_id", "s1_id", "y"]].assign(
            p=model.predict(_matrix(d, feats),
                            num_iteration=model.best_iteration).astype(np.float32)))
    return pd.concat(out, ignore_index=True)


def score_parts(st: StageRun, model: lgb.Booster, src, name: str, feats: list[str],
                with_y: bool = False) -> None:
    """Score every pair of a directory of feature parts, one part at a time; output
    ``name/`` has one part per input part, under the same file name (the rerank
    stage pairs them up by name), with q_id, s1_id[, y], p."""
    out_dir = st.path(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    n = n_pos = 0
    ys, ps = [], []
    for f in sorted(src.glob("*.parquet")):
        t = pd.read_parquet(f)
        pt = model.predict(_matrix(t, feats), num_iteration=model.best_iteration)
        keys = ["q_id", "s1_id"] + (["y"] if with_y else [])
        t[keys].assign(p=pt.astype(np.float32)).to_parquet(out_dir / f.name, index=False)
        n += len(t)
        if with_y:
            ys.append(t.y.to_numpy())
            ps.append(pt.astype(np.float32))
            n_pos += int(t.y.sum())
        del t
    st.register(name, out_dir, rows=n)
    if with_y and n:
        st.metric(f"{name}_auc", round(float(roc_auc_score(np.concatenate(ys),
                                                           np.concatenate(ps))), 6))


def score_test(st: StageRun, model: lgb.Booster, feats: list[str]) -> None:
    """Score every test pair, one feature part at a time."""
    with timed(st.log, "score test"):
        score_parts(st, model, st.input_path("features", "features_test"), "scores_test", feats)


def run_scorer(cfg: Config) -> None:
    with StageRun(cfg, "scorer") as st:
        if not cfg.run_train:  # test-only pass with the model of the train pass
            model = lgb.Booster(model_file=str(st.path("model.txt")))
            score_test(st, model, model.feature_name())
            return
        feat_dir = st.input_path("features", "features_train")
        first = pd.read_parquet(next(iter(sorted(feat_dir.glob("*.parquet"))))).head(1)
        feats = feature_columns(first, cfg.scorer.drop_features)
        hold_ids = None
        if cfg.rerank.hold_frac > 0:  # v2.4: held-out regions stay out of stage 1
            hold_ids = set(st.read_input("features_hold", "hold_queries",
                                         columns=["entity_id"]).entity_id.astype(str))
            st.metric("hold_queries_excluded", len(hold_ids))
        with timed(st.log, "load sampled train features"):
            S = load_sampled(feat_dir, cfg, st.log, feats, exclude=hold_ids)
        st.metric("rows", {"train": len(S["ytr"]), "val": len(S["yes"]),
                           "train_pos": int(S["ytr"].sum()), "val_pos": int(S["yes"].sum())})
        st.metric("n_features", len(feats))
        with timed(st.log, "train LightGBM"):
            # hand the matrices over, so they are freed once binned
            model = train_model(S.pop("Xtr"), S["ytr"], S.pop("Xes"), S["yes"], feats,
                                cfg, st.log)
        model.save_model(str(st.path("model.txt")))
        st.register("model", st.path("model.txt"))
        imp = pd.DataFrame({"feature": feats,
                            "gain": model.feature_importance("gain"),
                            "split": model.feature_importance("split")}
                           ).sort_values("gain", ascending=False)
        st.write_parquet(imp, "importance")
        st.log.info("top features by gain:\n%s", imp.head(40).to_string(index=False))

        del S
        with timed(st.log, "score all val pairs"):
            sv = predict_side(model, feat_dir, "val", feats)
        st.metric("best_iteration", model.best_iteration)
        st.metric("val_auc", round(float(roc_auc_score(sv.y, sv.p)), 6))
        st.metric("val_logloss", round(float(log_loss(sv.y, sv.p)), 6))
        st.write_parquet(sv, "scores_val")
        del sv
        if hold_ids is not None:  # held-out regions: out-of-sample p1 for stage 2
            with timed(st.log, "score all hold pairs"):
                score_parts(st, model, st.input_path("features_hold", "features_hold"),
                            "scores_hold", feats, with_y=True)

        if cfg.run_test:
            score_test(st, model, feats)
