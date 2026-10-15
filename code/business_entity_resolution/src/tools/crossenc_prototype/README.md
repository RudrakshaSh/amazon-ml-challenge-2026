# Cross-encoder prototype scripts (where the submitted file came from)

We used these three scripts to fine-tune and score the cross-encoder behind the submitted `output/matching_results.tsv` (v2.7, first run, 2026-09-27; public leaderboard 0.98316). The training logic is unchanged. We only replaced our machine's file paths with configurable ones (`paths.py`) and edited one comment.

Everything after that (re-ranking, the decision and writing the submission) ran with this package's own code (`python -m ber rerank / decide / submit`).

| Script | What it does |
|---|---|
| `prep.py` | Picks the training pairs: training-side records outside the held-out regions that stage 1 didn't train on (`stable_unit(q_id, seed + 4) >= 0.14`), kept at p1 ≥ 0.001. That gives 982,265 pairs, 68 % of them true. It also writes the record texts (`name core + legal ; address + region`) for those pairs and for every stage-2 pair (val, hold, test), in stage 2's row order |
| `train.py` | Fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22.7 M parameters): 2 epochs, batch 256, AdamW at lr 5e-5, weight decay 0.01, 5 % linear warm-up, fp16, at most 64 tokens, binary cross-entropy on the match label |
| `score.py` | Scores every stage-2 pair and writes `<out>/{val,hold,test_France,test_India,test_US}.parquet` (q_id, s1_id, ce) |

**Running them yourself.** Run everything from the project root (the directory with `dataset/`). `paths.py` sets where the scripts read and write; each location can be changed with an environment variable:

| Variable | Default | What it is |
|---|---|---|
| `BER_ARTIFACTS` | `artifacts` | the pipeline's artifacts directory |
| `CE_RUN_FEATURES` | `main` | run with the `normalize` and `features` outputs |
| `CE_RUN_SCORER` | `main` | run with the `scorer` and `features-hold` outputs |
| `CE_RUN_RERANK` | `main` | run with a stage-2 pass *without* the cross-encoder (fixes the row order) |
| `CE_WORK` | `artifacts/crossenc_prototype` | where the pair texts, the model and the scores go |
| `CE_BASE_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | the Hugging Face id, or a local copy |

Our own runs used separate run ids: `v2` for normalize and features, `v25` for the scorer and features-hold, and `v25b` for stage 2 without the cross-encoder.

The scripts need stage-2 outputs to exist first, so the order is:

```bash
export PYTHONPATH="$PWD/code/business_entity_resolution/src" PYTHONUTF8=1
P=code/business_entity_resolution/src/tools/crossenc_prototype

# 1. the pipeline up to stage 2, without the cross-encoder
for s in split normalize candidates features features-hold scorer; do python -m ber $s --run main || break; done
python -m ber rerank --run main --set rerank.extra_scores=null

# 2. the prototype: pair texts, fine-tuning (2 epochs), scoring
python $P/prep.py && python $P/train.py 2 && python $P/score.py

# 3. stage 2 again with the cross-encoder scores, then the decision and the submission
S=artifacts/crossenc_prototype/scores
python -m ber rerank --run main --set rerank.extra_scores=$S
python -m ber decide --run main --set rerank.extra_scores=$S
python -m ber submit --run main --set rerank.extra_scores=$S
```

**How they relate to the pipeline's `crossenc` stage** (`src/ber/crossenc/stage.py`):
- **It's the same logic, packaged as a stage.** Pair selection is the same (982,265 training pairs, 68.0 % true, in both), and so are the text, the model and the hyper-parameters. The default config runs it (`rerank.extra_scores: crossenc`).
- **There's one small difference.** `train.py` draws a 200,000-pair validation sample from its random generator before training, just to print an AUC at the end of each epoch. That changes its batch order.
- **Every run comes out slightly different anyway,** because GPU fine-tuning isn't bit-for-bit reproducible:

  | Run | Validation macro F0.5 | Public LB |
  |---|---:|---:|
  | **These scripts (submitted)** | **0.99078** | **0.98316** |
  | `crossenc` stage re-run | 0.99060 | 0.982681 |

- **Why we submitted this run:** it's better on the labelled validation set (a paired per-entity difference of +0.00018, z ≈ 3.9), and the public leaderboard agrees.
