"""Where the prototype scripts read and write.

Every location can be changed with an environment variable. The defaults assume you run
from the project root (the directory with dataset/ and artifacts/) and that the pipeline
stages ran under the run id "main". Our own runs used separate ids: v2 (normalize,
features), v25 (scorer, features-hold) and v25b (stage 2 without the cross-encoder).
"""
import os
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]  # code/business_entity_resolution/src
ARTIFACTS = Path(os.environ.get("BER_ARTIFACTS", "artifacts"))
RUN_FEATURES = os.environ.get("CE_RUN_FEATURES", "main")  # normalize + features
RUN_SCORER = os.environ.get("CE_RUN_SCORER", "main")      # scorer + features-hold
RUN_RERANK = os.environ.get("CE_RUN_RERANK", "main")      # stage 2 without the cross-encoder (row order)
WORK = Path(os.environ.get("CE_WORK", str(ARTIFACTS / "crossenc_prototype")))
DATA = WORK / "data"      # prep.py output: pair texts
MODEL = WORK / "model"    # train.py output: the fine-tuned model
SCORES = WORK / "scores"  # score.py output: what rerank.extra_scores points at
BASE_MODEL = os.environ.get("CE_BASE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
