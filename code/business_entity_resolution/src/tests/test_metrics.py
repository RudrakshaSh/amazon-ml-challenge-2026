import importlib.util
import random
from pathlib import Path

import pytest

from ber.eval.metrics import f05, macro_f05, per_entity, summarize

ROOT = Path(__file__).resolve().parents[4]  # the directory with dataset/ and utils/


@pytest.fixture(scope="module")
def reference():
    if not (ROOT / "utils" / "score.py").is_file():  # the organisers' scorer, next to code/
        pytest.skip("utils/score.py not found (run from the student_resource directory)")
    spec = importlib.util.spec_from_file_location("score", ROOT / "utils" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_example():
    truth = {"S1-1": ["S2-47", "S3-812"]}
    pred = {"S1-1": ["S2-47", "S2-193", "S3-812"]}
    assert macro_f05(truth, pred) == pytest.approx(0.714285, abs=1e-5)


@pytest.mark.parametrize("tp,npred,ntrue,expected", [
    (0, 0, 0, 1.0), (0, 1, 0, 0.0), (0, 0, 2, 0.0), (0, 3, 2, 0.0), (2, 2, 2, 1.0)])
def test_edge_cases(tp, npred, ntrue, expected):
    assert f05(tp, npred, ntrue) == expected


def test_matches_utils_score_on_random_cases(reference):
    rng = random.Random(0)
    ids = [f"S2-{i}" for i in range(12)]
    truth, pred = {}, {}
    for k in range(3000):
        truth[f"S1-{k}"] = set(rng.sample(ids, rng.randint(0, 4)))
        pred[f"S1-{k}"] = set(rng.sample(ids, rng.randint(0, 4)))
    df = per_entity(truth, pred)
    ref = [reference.f_half(truth[k], pred[k])[0] for k in truth]
    assert df.f05.tolist() == pytest.approx(ref)


def test_missing_predictions_count_as_empty_and_summary():
    truth = {"a": ["S2-1"], "b": [], "c": ["S3-1", "S3-2"]}
    df = per_entity(truth, {"c": ["S3-1"]})
    assert df.f05.tolist()[:2] == [0.0, 1.0]
    s = summarize(df)
    assert s.loc["ALL", "n"] == 3 and s.loc["ALL", "singleton_acc"] == 1.0
    assert s.loc["ALL", "pair_p"] == 1.0 and s.loc["ALL", "pair_r"] == pytest.approx(1 / 3)
