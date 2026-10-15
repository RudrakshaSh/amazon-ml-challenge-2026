"""Fine-tune a MiniLM cross-encoder on record pairs (exploration: a transformer re-scorer).

Input text per pair: "[CLS] record text [SEP] S1 text [SEP]" (texts from prep.py), max 64
tokens; binary cross-entropy on the match label. Trained on stage 1's own regions (records
stage 1 did not train on), so its scores are out-of-sample on val, hold and test.
usage: python train.py [epochs] [out_dir]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ARTIFACTS, BASE_MODEL as BASE, DATA, MODEL, RUN_RERANK  # noqa: E402

EPOCHS, OUT = 2, MODEL  # overridden by argv in main()
MAX_LEN, BS, LR, WD, WARMUP, SEED = 64, 256, 5e-5, 0.01, 0.05, 42
N_VAL = 200_000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def encode(tok, a, b):
    """Padded int32 ids, int8 token types and the true lengths (for trimming batches)."""
    enc = tok(list(a), list(b), truncation="longest_first", max_length=MAX_LEN,
              padding="max_length", return_attention_mask=True, return_token_type_ids=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    tt = np.asarray(enc["token_type_ids"], dtype=np.int8)
    ln = np.asarray(enc["attention_mask"], dtype=np.int8).sum(1).astype(np.int16)
    return ids, tt, ln


def batches(ids, tt, ln, idx, bs, dev):
    for a in range(0, len(idx), bs):
        j = idx[a:a + bs]
        m = int(ln[j].max())
        x = torch.from_numpy(ids[j, :m].astype(np.int64)).to(dev, non_blocking=True)
        t = torch.from_numpy(tt[j, :m].astype(np.int64)).to(dev, non_blocking=True)
        att = (torch.arange(m, device=dev)[None, :] < torch.from_numpy(ln[j].astype(np.int64)).to(dev)[:, None]).long()
        yield j, x, t, att


@torch.no_grad()
def predict(model, ids, tt, ln, dev, bs=512):
    """Logits in the input order; batches of similar length; halves the batch on a CUDA
    out-of-memory error (GPU memory is shared with the training state)."""
    model.eval()
    torch.cuda.empty_cache()
    out = np.empty(len(ids), dtype=np.float32)
    order = np.argsort(ln, kind="stable")  # similar lengths per batch: less padding
    a = 0
    while a < len(order):
        j = order[a:a + bs]
        try:
            for jj, x, t, att in batches(ids, tt, ln, j, bs, dev):
                with torch.autocast("cuda", dtype=torch.float16):
                    out[jj] = model(input_ids=x, token_type_ids=t,
                                    attention_mask=att).logits.float().squeeze(-1).cpu().numpy()
            a += bs
        except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            if "out of memory" not in str(e) or bs <= 32:
                raise
            torch.cuda.empty_cache()
            bs //= 2
            log(f"CUDA out of memory: inference batch -> {bs}")
    model.train()
    return out


def main():
    global EPOCHS, OUT
    if len(sys.argv) > 1:
        EPOCHS = int(sys.argv[1])
    if len(sys.argv) > 2:
        OUT = Path(sys.argv[2])
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    tr = pd.read_parquet(DATA / "train.parquet", columns=["q_text", "s_text", "y", "p1"])
    log(f"train pairs {len(tr):,} (positive rate {tr.y.mean():.3f}); tokenising")
    ids, tt, ln = encode(tok, tr.q_text, tr.s_text)
    y = tr.y.to_numpy(np.float32)
    log(f"lengths: median {np.median(ln):.0f}, 99% {np.quantile(ln, 0.99):.0f}, "
        f"truncated {(ln >= MAX_LEN).mean():.4f}")
    va = pd.read_parquet(DATA / "val.parquet", columns=["q_text", "s_text", "y"])
    sv = pd.read_parquet(ARTIFACTS / RUN_RERANK / "rerank" / "scores_val.parquet",
                         columns=["y", "p", "p1"])
    assert (sv.y.to_numpy() == va.y.to_numpy()).all()
    vi = np.sort(rng.choice(len(va), size=min(N_VAL, len(va)), replace=False))
    vids, vtt, vln = encode(tok, va.q_text.iloc[vi], va.s_text.iloc[vi])
    vy, vp1, vp2 = va.y.to_numpy()[vi], sv.p1.to_numpy()[vi], sv.p.to_numpy()[vi]
    log(f"val sample {len(vi):,}: AUC p1 {roc_auc_score(vy, vp1):.6f}, p2 {roc_auc_score(vy, vp2):.6f}")
    del tr, va, sv

    steps = EPOCHS * math.ceil(len(y) / BS)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": WD},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(y))
        run = 0.0
        for j, x, t, att in batches(ids, tt, ln, perm, BS, dev):
            yy = torch.from_numpy(y[j]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
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
            if step % 500 == 0:
                log(f"epoch {ep} step {step}/{steps} loss {run:.4f} ({step / (time.time() - t0):.1f} steps/s)")
        ck = OUT / f"epoch{ep}"
        ck.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ck)
        tok.save_pretrained(ck)
        pv = predict(model, vids, vtt, vln, dev)
        pr = 1 / (1 + np.exp(-pv))
        log(f"epoch {ep}: val sample AUC {roc_auc_score(vy, pv):.6f} logloss "
            f"{log_loss(vy, np.clip(pr, 1e-7, 1 - 1e-7)):.5f}; AUC of p2 + CE rank-avg "
            f"{roc_auc_score(vy, pd.Series(vp2).rank().to_numpy() + pd.Series(pv).rank().to_numpy()):.6f}")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
