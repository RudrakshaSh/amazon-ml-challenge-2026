"""``ber submit`` (task T3.5): write output/matching_results.tsv for the test
set and run the official validator on it.

With ``submit.write_candidates`` also output/candidate_pairs.tsv, required in the
final package: "the exact set of records you feed into your matching model for
inference ... if your pipeline has several blocking/filtering stages, the last
one". That is the pair set of the scores ``decide`` consumed
(``decide.scores_stage``): with the stage-2 re-ranker, the pairs that passed the
stage-1 filter (p1 >= rerank.min_p1, ~3 per record); with the stage-1 scorer
alone, every TF-IDF candidate (~2.4e8 pairs, ~3 GB). Matches are a subset by
construction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import StageRun
from .config import Config
from .io import CANDIDATE_HEADER, MATCHING_HEADER
from .log import timed


def _write_grouped(path, s1_ids: np.ndarray, s1_codes: np.ndarray, q_ids: np.ndarray,
                   q_codes: np.ndarray, header: tuple[str, str]) -> int:
    """One line per S1 (all of ``s1_ids``, in order), listing its q ids."""
    order = np.lexsort((q_codes, s1_codes))
    s_sorted, q_sorted = s1_codes[order], q_codes[order]
    starts = np.searchsorted(s_sorted, np.arange(len(s1_ids)), side="left")
    ends = np.searchsorted(s_sorted, np.arange(len(s1_ids)), side="right")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for i, sid in enumerate(s1_ids):
            qs = q_sorted[starts[i]:ends[i]]
            f.write(f"{sid}\t{','.join(q_ids[qs])}\n")
    return len(s1_ids)


def run_submit(cfg: Config) -> None:
    if cfg.data.slice is not None:
        raise SystemExit("submit needs the full pipeline (data.slice must be null)")
    with StageRun(cfg, "submit") as st:
        recs = pd.read_parquet(st.input_path("normalize", "records_test"),
                               columns=["entity_id", "source"])
        s1_ids = recs.entity_id[recs.source == 1].to_numpy()
        q_ids = recs.entity_id[recs.source != 1].to_numpy()
        s1_cat = pd.CategoricalDtype(s1_ids)
        q_cat = pd.CategoricalDtype(q_ids)
        out = cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)

        dec = st.read_input("decide", "decisions_test")
        s_codes = pd.Categorical(dec.s1_id.astype(str), dtype=s1_cat).codes
        q_codes = pd.Categorical(dec.q_id.astype(str), dtype=q_cat).codes
        if (s_codes < 0).any() or (q_codes < 0).any():
            raise ValueError("decisions reference ids that are not in the test set")
        n = _write_grouped(out / "matching_results.tsv", s1_ids, s_codes, q_ids, q_codes,
                           MATCHING_HEADER)
        st.register("matching_results", out / "matching_results.tsv", rows=n)
        st.metric("matched_pairs", len(dec))

        if cfg.submit.write_candidates:
            _write_candidates(st, cfg, out, s1_ids, q_ids, s1_cat, q_cat)
        else:
            st.log.info("candidate_pairs.tsv skipped (submit.write_candidates=false)")

        with timed(st.log, "official validator"):
            _validate(st, cfg, out)


def _write_candidates(st, cfg: Config, out, s1_ids, q_ids, s1_cat, q_cat) -> None:
    with timed(st.log, "candidate_pairs.tsv"):
        sc, qc = [], []
        src = st.input_path(cfg.decide.scores_stage, "scores_test")
        for f in sorted(src.glob("*.parquet")):
            p = pd.read_parquet(f, columns=["q_id", "s1_id"])
            sc.append(pd.Categorical(p.s1_id.astype(str), dtype=s1_cat).codes)
            qc.append(pd.Categorical(p.q_id.astype(str), dtype=q_cat).codes)
        sc_all, qc_all = np.concatenate(sc), np.concatenate(qc)
        n = _write_grouped(out / "candidate_pairs.tsv", s1_ids, sc_all, q_ids, qc_all,
                           CANDIDATE_HEADER)
        st.register("candidate_pairs", out / "candidate_pairs.tsv", rows=n)
        st.metric("candidate_pairs", len(sc_all))


def _validate(st, cfg: Config, out) -> None:
    """Official validator. It runs from utils/ on purpose: when --candidate is
    omitted the validator loads ./output/candidate_pairs.tsv if it exists, and at
    ~2.4e8 ids (stage-1 candidates) that needs ~30 GB of RAM. The candidate file
    is checked only when it is the small stage-2 input (``decide.scores_stage``
    is not the stage-1 scorer)."""
    cmd = [sys.executable, "validate_submission.py",
           "--matching", str(out / "matching_results.tsv"),
           "--test-dir", str(cfg.data_dir / "test")]
    cand = out / "candidate_pairs.tsv"
    if cfg.submit.write_candidates and cfg.decide.scores_stage != "scorer" and cand.exists():
        cmd += ["--candidate", str(cand)]
    # the organisers' utils/ next to dataset/, else the copy shipped in src/tools/
    vdir = cfg.root / "utils"
    if not (vdir / "validate_submission.py").exists():
        vdir = Path(__file__).resolve().parents[1] / "tools"
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=vdir)
    st.log.info("validator output:\n%s%s", res.stdout, res.stderr)
    st.metric("validator", "PASS" if res.returncode == 0 else "FAIL")
    if res.returncode != 0:
        raise RuntimeError("submission failed validation; see log.txt")
