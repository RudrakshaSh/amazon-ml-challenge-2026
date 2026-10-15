"""Score the stage-2 pairs with a fine-tuned cross-encoder; one output file per input file,
in the same row order (so stage 2 can align them with its own rows).
usage: python score.py [model_dir] [out_dir]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, MODEL as DEFAULT_MODEL, SCORES  # noqa: E402
from train import encode, log, predict  # noqa: E402

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SCORES
CHUNK = 1_000_000
NAMES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["val", "hold", "test_France", "test_India", "test_US"]


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        d = pd.read_parquet(DATA / f"{name}.parquet", columns=["q_id", "s1_id", "q_text", "s_text"])
        ce = np.empty(len(d), dtype=np.float32)
        t0 = time.time()
        for a in range(0, len(d), CHUNK):
            ids, tt, ln = encode(tok, d.q_text.iloc[a:a + CHUNK], d.s_text.iloc[a:a + CHUNK])
            ce[a:a + CHUNK] = predict(model, ids, tt, ln, dev, bs=2048)
        pd.DataFrame({"q_id": d.q_id.to_numpy(), "s1_id": d.s1_id.to_numpy(), "ce": ce}).to_parquet(
            OUT / f"{name}.parquet", index=False)
        log(f"{name}: {len(d):,} pairs in {time.time() - t0:.0f}s")
        del d, ce


if __name__ == "__main__":
    main()
