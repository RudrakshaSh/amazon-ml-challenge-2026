"""``ber crossenc``: a transformer cross-encoder next to the GBMs.

A pretrained MiniLM cross-encoder (``crossenc.base_model``, Apache-2.0) reads both records'
normalised text jointly ("name core + legal form ; address + region" of the S2/S3 record
and of the S1) and is fine-tuned to predict the match. Its logit and the logit's context
(margin and rank among the record's candidates and within the S1's group) become stage-2
features (``rerank.extra_scores: crossenc``). On its own it is a weak matcher (val AUC
0.975 on the stage-2 pairs vs 0.9989 for stage 2), but its view of which candidate wins
is new evidence: val F0.5 0.98918 -> 0.99078 with it.

Training pairs come from stage 1's own regions: train-side records outside the hold
regions that stage 1 did *not* train on (``stable_unit(q_id, seed + 4) >=
scorer.train_query_frac``, the scorer's sampling rule), so their stage-1 p is
out-of-sample; they are kept at p1 >= ``rerank.min_p1`` exactly like stage 2's inputs.
Stage 2 trains on val + hold and runs on test, so the cross-encoder is out-of-sample on
every pair stage 2 ever sees.

Outputs (q_id, s1_id, ce) in stage 2's row order: ``ce_val``, ``ce_hold`` (train pass),
``ce_test_<country>`` (test pass), and the fine-tuned model directory ``model``.
torch and transformers are imported here only (the rest of the pipeline does not need them).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed


def texts(path, ids) -> pd.Series:
    """entity_id -> "name core legal ; address region" for the given ids of a
    ``records_<split>`` directory."""
    d = ds.dataset(path, format="parquet")
    arr = pa.array(sorted(set(ids)), type=pa.large_string())
    t = d.to_table(columns=["entity_id", "name_core", "name_legal", "addr_norm", "region"],
                   filter=ds.field("entity_id").isin(arr))
    df = t.to_pandas()
    name = (df.name_core + " " + df.name_legal).str.strip()
    addr = (df.addr_norm + " " + df.region).str.strip()
    return pd.Series((name + " ; " + addr).to_numpy(), index=df.entity_id.to_numpy())


def with_text(pairs: pd.DataFrame, rec_path) -> pd.DataFrame:
    tx = texts(rec_path, np.concatenate([pairs.q_id.to_numpy(), pairs.s1_id.to_numpy()]))
    out = pairs.assign(q_text=tx.reindex(pairs.q_id.to_numpy()).to_numpy(),
                       s_text=tx.reindex(pairs.s1_id.to_numpy()).to_numpy())
    if out.q_text.isna().any() or out.s_text.isna().any():
        raise RuntimeError("crossenc: records missing for some pair ids")
    return out


def train_pairs(st: StageRun, cfg: Config) -> pd.DataFrame:
    """Stage-1 pool pairs with out-of-sample p1 >= rerank.min_p1 (see module doc)."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(st.input_path("scorer", "model")))
    feats = model.feature_name()
    hold = None
    if cfg.rerank.hold_frac > 0:
        h = st.read_input("features_hold", "hold_queries", columns=["entity_id"]).entity_id
        hold = pa.array(sorted(h.astype(str)), type=pa.large_string())
    rows, n_seen = [], 0
    for f in sorted(st.input_path("features", "features_train").glob("*.parquet")):
        t = pq.read_table(f, columns=["q_id", "side"])
        m = pc.equal(t.column("side"), "train").to_numpy(zero_copy_only=False)
        if not m.any():
            continue
        m &= stable_unit(t.column("q_id").to_pandas().astype(str), cfg.seed + 4) \
            >= cfg.scorer.train_query_frac
        if hold is not None:
            m &= ~pc.is_in(t.column("q_id").cast(pa.large_string()),
                           value_set=hold).to_numpy(zero_copy_only=False)
        if not m.any():
            continue
        d = pd.read_parquet(f, columns=["q_id", "s1_id", "y"] + feats)[m]
        n_seen += len(d)
        p = model.predict(np.ascontiguousarray(d[feats].to_numpy(dtype=np.float32)))
        k = p >= cfg.rerank.min_p1
        rows.append(pd.DataFrame({"q_id": d.q_id.astype(str).to_numpy()[k],
                                  "s1_id": d.s1_id.astype(str).to_numpy()[k],
                                  "y": d.y.to_numpy()[k].astype(np.int8)}))
        del d, t
    out = pd.concat(rows, ignore_index=True)
    st.metric("train_pairs", {"scored": n_seen, "kept": len(out), "records": int(out.q_id.nunique()),
                              "positive_rate": round(float(out.y.mean()), 4)})
    return out


def kept_side(st: StageRun, cfg: Config, side: str) -> pd.DataFrame:
    """Val or hold pairs with p1 >= min_p1, in the order stage 2 loads them."""
    if side == "val":
        t = pq.read_table(st.input_path("scorer", "scores_val"), columns=["q_id", "s1_id", "p"])
        tables = [t]
    else:
        tables = [pq.read_table(f, columns=["q_id", "s1_id", "p"])
                  for f in sorted(st.input_path("scorer", "scores_hold").glob("*.parquet"))]
    out = []
    for t in tables:
        k = t.column("p").to_numpy() >= cfg.rerank.min_p1
        out.append(pd.DataFrame({"q_id": t.column("q_id").to_numpy(zero_copy_only=False)[k].astype(str),
                                 "s1_id": t.column("s1_id").to_numpy(zero_copy_only=False)[k].astype(str)}))
    return pd.concat(out, ignore_index=True)


def kept_test(st: StageRun, cfg: Config):
    """(country, test pairs with p1 >= min_p1) in the order stage 2's test pass builds
    them: score parts grouped by the country of their first S1, countries sorted."""
    rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                         columns=["entity_id", "country"], filters=[("source", "==", 1)])
    s1_country = rt.set_index("entity_id").country
    by_country: dict[str, list] = {}
    for f in sorted(st.input_path("scorer", "scores_test").glob("*.parquet")):
        first = pq.read_table(f, columns=["s1_id"]).column("s1_id").slice(0, 1).to_pylist()
        if first:
            by_country.setdefault(s1_country[first[0]], []).append(f)
    for country, files in sorted(by_country.items()):
        out = []
        for f in files:
            t = pq.read_table(f, columns=["q_id", "s1_id", "p"])
            k = t.column("p").to_numpy() >= cfg.rerank.min_p1
            out.append(pd.DataFrame({"q_id": t.column("q_id").to_numpy(zero_copy_only=False)[k].astype(str),
                                     "s1_id": t.column("s1_id").to_numpy(zero_copy_only=False)[k].astype(str)}))
        yield country, pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ model

def encode(tok, a, b, max_len: int):
    """Padded int32 ids, int8 token types and true lengths (to trim each batch)."""
    enc = tok(list(a), list(b), truncation="longest_first", max_length=max_len,
              padding="max_length", return_attention_mask=True, return_token_type_ids=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    tt = np.asarray(enc["token_type_ids"], dtype=np.int8)
    ln = np.asarray(enc["attention_mask"], dtype=np.int8).sum(1).astype(np.int16)
    return ids, tt, ln


def _batches(ids, tt, ln, idx, bs, dev):
    import torch
    for a in range(0, len(idx), bs):
        j = idx[a:a + bs]
        m = int(ln[j].max())
        x = torch.from_numpy(ids[j, :m].astype(np.int64)).to(dev, non_blocking=True)
        t = torch.from_numpy(tt[j, :m].astype(np.int64)).to(dev, non_blocking=True)
        lens = torch.from_numpy(ln[j].astype(np.int64)).to(dev)
        att = (torch.arange(m, device=dev)[None, :] < lens[:, None]).long()
        yield j, x, t, att


def predict(model, ids, tt, ln, dev, bs: int, log=None) -> np.ndarray:
    """Logits in input order; batches of similar length; the batch is halved on a CUDA
    out-of-memory error."""
    import torch
    model.eval()
    torch.cuda.empty_cache()
    out = np.empty(len(ids), dtype=np.float32)
    order = np.argsort(ln, kind="stable")
    a = 0
    with torch.no_grad():
        while a < len(order):
            j = order[a:a + bs]
            try:
                for jj, x, t, att in _batches(ids, tt, ln, j, bs, dev):
                    with torch.autocast("cuda", dtype=torch.float16):
                        out[jj] = model(input_ids=x, token_type_ids=t, attention_mask=att
                                        ).logits.float().squeeze(-1).cpu().numpy()
                a += bs
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e) or bs <= 32:
                    raise
                torch.cuda.empty_cache()
                bs //= 2
                if log:
                    log.info("CUDA out of memory: inference batch -> %d", bs)
    model.train()
    return out


def fine_tune(cfg: Config, pairs: pd.DataFrame, out_dir, log) -> None:
    """Fine-tune ``crossenc.base_model`` on (q_text, s_text, y) with binary cross-entropy
    (AdamW, linear warm-up and decay, fp16 autocast); saves model + tokenizer to out_dir."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    cc = cfg.crossenc
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cc.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(cc.base_model, num_labels=1).to(dev)
    ids, tt, ln = encode(tok, pairs.q_text, pairs.s_text, cc.max_len)
    y = pairs.y.to_numpy(np.float32)
    log.info("crossenc: %d training pairs, median length %d tokens, %.4f truncated",
             len(y), int(np.median(ln)), float((ln >= cc.max_len).mean()))
    steps = cc.epochs * math.ceil(len(y) / cc.batch_size)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": cc.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cc.learning_rate)
    sched = get_linear_schedule_with_warmup(opt, int(cc.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(cc.epochs):
        for j, x, t, att in _batches(ids, tt, ln, rng.permutation(len(y)), cc.batch_size, dev):
            yy = torch.from_numpy(y[j]).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(input_ids=x, token_type_ids=t, attention_mask=att).logits.squeeze(-1)
            loss = lossf(logit.float(), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * float(loss.detach()) if step > 1 else float(loss.detach())
            if step % 1000 == 0:
                log.info("crossenc: epoch %d step %d/%d loss %.4f (%.1f steps/s)", ep, step, steps,
                         run, step / (time.time() - t0))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def score(cfg: Config, model_dir, pairs: pd.DataFrame, log) -> np.ndarray:
    """Cross-encoder logits for (q_text, s_text), in the pairs' order."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cc = cfg.crossenc
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    ce = np.empty(len(pairs), dtype=np.float32)
    for a in range(0, len(pairs), cc.chunk):
        ids, tt, ln = encode(tok, pairs.q_text.iloc[a:a + cc.chunk], pairs.s_text.iloc[a:a + cc.chunk],
                             cc.max_len)
        ce[a:a + cc.chunk] = predict(model, ids, tt, ln, dev, cc.infer_batch, log)
    del model
    torch.cuda.empty_cache()
    return ce


def run_crossenc(cfg: Config) -> None:
    with StageRun(cfg, "crossenc") as st:
        model_dir = st.path("model")
        if cfg.run_train:
            rec = st.input_path("normalize", "records_train")
            with timed(st.log, "training pairs (stage 1's own out-of-sample records)"):
                tr = with_text(train_pairs(st, cfg), rec)
            with timed(st.log, "fine-tune"):
                fine_tune(cfg, tr, model_dir, st.log)
            del tr
            st.register("model", model_dir)
            sides = ["val", "hold"] if cfg.rerank.hold_frac > 0 else ["val"]
            for side in sides:
                with timed(st.log, f"score {side} pairs"):
                    kp = kept_side(st, cfg, side)
                    ce = score(cfg, model_dir, with_text(kp, rec), st.log)
                    st.write_parquet(kp.assign(ce=ce), f"ce_{side}")
                    if side == "val":
                        y = pd.read_parquet(st.input_path("scorer", "scores_val"), columns=["y", "p"])
                        y = y.y.to_numpy()[y.p.to_numpy() >= cfg.rerank.min_p1]
                        from sklearn.metrics import roc_auc_score
                        st.metric("val_auc", round(float(roc_auc_score(y, ce)), 6))
        if cfg.run_test:
            rec = st.input_path("normalize", "records_test")
            for country, kp in kept_test(st, cfg):
                with timed(st.log, f"score test {country} pairs"):
                    ce = score(cfg, model_dir, with_text(kp, rec), st.log)
                    st.write_parquet(kp.assign(ce=ce), f"ce_test_{country}")
