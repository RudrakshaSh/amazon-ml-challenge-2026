import numpy as np
import pandas as pd
import pytest

from ber.candidates.generate import generate_country
from ber.candidates.tfidf import fit_index, pair_cosine, ranks_within_query, topk
from ber.config import load_config
from ber.log import get_logger


def test_ranks_within_query():
    assert ranks_within_query(np.array([0, 0, 0, 3, 3, 7])).tolist() == [0, 1, 2, 0, 1, 0]


def test_topk_finds_exact_match_first():
    vec, S = fit_index(["alpha beta", "gamma delta", "alpha gamma"], 3, 1.0)
    Q = vec.transform(["alpha beta", "delta gamma"]).tocsr()
    qi, sj, sim = topk(Q, S, 2, chunk=1, n_threads=1)
    assert sj[qi == 0][0] == 0 and sj[qi == 1][0] == 1
    assert np.allclose(pair_cosine(Q, S, qi, sj), sim, atol=1e-5)


@pytest.mark.parametrize("by_region", [False, True])
def test_generate_country(by_region):
    s1 = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "source": 1, "country": "US",
                       "name_core": ["vision health", "prairie grand", "vision care"],
                       "addr_bag": ["7622 arbory ct laurel", "15 dr mountain view wolcott",
                                    "1 main st laurel"],
                       "region": ["MD", "CT", "MD"], "name_indic": False})
    q = pd.DataFrame({"entity_id": ["S2-1", "S3-2", "S3-3"], "source": [2, 3, 3], "country": "US",
                      "name_core": ["vision heaheh", "prairie 6rand", "vision health"],
                      "addr_bag": ["arbory ct laurel", "15 dr mountain view wolcott", ""],
                      "region": ["MD", "CT", ""], "name_indic": False})
    cfg = load_config(overrides=["candidates.k_name=2", "candidates.k_addr=2",
                                 "candidates.k_combo=2", "candidates.n_threads=1", "candidates.k_name_noaddr=2",
                                 f"candidates.block_by_region={str(by_region).lower()}"])
    c = pd.concat(list(generate_country(s1, q, cfg, get_logger("test"),
                                        {"MD": ["MD"], "CT": ["CT"]}))).astype(
        {"q_id": str, "s1_id": str})
    best = c.sort_values("cos_combo", ascending=False).groupby("q_id").s1_id.first()
    assert best.to_dict() == {"S2-1": "S1-a", "S3-2": "S1-b", "S3-3": "S1-a"}
    assert set(c.columns) >= {"cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr",
                              "rank_combo"}
    assert c.rank_name.max() <= 2 and not c.duplicated(["q_id", "s1_id"]).any()
    if by_region:  # S3-2 (CT) must not see MD entities
        assert set(c.s1_id[c.q_id == "S3-2"]) == {"S1-b"}


def test_region_neighbors_symmetric_and_thresholded():
    from ber.candidates.generate import region_neighbors
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "source": 1, "country": "India",
                       "region": ["TG", "MH", "AP"], "owner": ""})
    q = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(12)], "source": 2,
                      "country": "India",
                      "region": ["AP"] * 5 + ["TG"] * 5 + ["MH", "MH"],
                      "owner": ["S1-1"] * 4 + ["S1-3"] + ["S1-1"] * 5 + ["S1-2", "S1-2"]})
    nb = region_neighbors(pd.concat([s1, q]), 0.1)["India"]
    assert nb["AP"] == ["AP", "TG"] and nb["TG"] == ["AP", "TG"] and nb["MH"] == ["MH"]


def test_gpu_topk_matches_cpu():
    import numpy as np
    import pytest
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    from ber.candidates.tfidf import gpu_available, topk
    if not gpu_available():
        pytest.skip("no CUDA device / CuPy")
    rng = np.random.default_rng(0)
    S = normalize(sp.random(3000, 400, density=0.03, format="csr", random_state=1,
                            dtype=np.float32))
    Q = normalize(sp.random(500, 400, density=0.03, format="csr", random_state=2,
                            dtype=np.float32))
    a = topk(Q, S, 5, 1000, 2)
    b = topk(Q, S, 5, 1000, 2, device="gpu")
    assert (a[0] == b[0]).all()                       # same number of hits per query
    np.testing.assert_allclose(a[2], b[2], atol=1e-5)  # same similarities in rank order
    del rng
