import numpy as np
import pandas as pd
import pytest

from ber.decide.expected_f import best_prefix, select
from ber.decide.stage import decide, one_owner


def test_best_prefix_prefers_empty_when_unlikely():
    assert best_prefix(np.array([0.1]))[0] == 0          # P(no match)=0.9 beats F of keeping
    assert best_prefix(np.array([0.95, 0.9, 0.2]))[0] == 2
    assert best_prefix(np.array([]))[0] == 0


def test_best_prefix_value_matches_certain_case():
    k, ef = best_prefix(np.array([1.0, 1.0]))
    assert k == 2 and ef == pytest.approx(1.0)


def test_one_owner_and_select():
    scores = pd.DataFrame({"q_id": ["a", "a", "b", "c"], "s1_id": ["X", "Y", "X", "Z"],
                           "p": [0.9, 0.95, 0.8, 0.05]})
    a = one_owner(scores)
    assert dict(zip(a.q_id, a.s1_id)) == {"a": "Y", "b": "X", "c": "Z"}
    kept = select(a)
    assert set(kept.q_id) == {"a", "b"}
    assert set(decide(a, "threshold", 0.5, 0, 1).q_id) == {"a", "b"}
    with pytest.raises(ValueError):
        decide(a, "magic", 0, 0, 1)


def test_tie_duplicates():
    from ber.decide.stage import dup_keys, tie_duplicates
    recs = pd.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"], "source": [2, 2, 2],
                         "country": "US", "name_raw": ["Acme", "Acme", "Other"],
                         "addr_raw": ["1 Main St", "1 Main St", "1 Main St"]})
    a = pd.DataFrame({"q_id": ["S2-1", "S2-2", "S2-3"], "s1_id": ["X", "Y", "Z"],
                      "p": [0.4, 0.9, 0.5]})
    t = tie_duplicates(a, dup_keys(recs))
    assert t.s1_id.tolist() == ["Y", "Y", "Z"] and t.p.tolist() == [0.9, 0.9, 0.5]


def test_anchor_name_kinds():
    from collections import Counter
    from ber.decide.anchor import name_kinds
    s1 = ["ski college"] * 50 + ["ski comite"] * 50 + ["lepicure club", "zyx"] + ["club x"] * 100
    df = Counter(t for n in s1 for t in set(n.split()))
    q = ["college ski", "ski colege", "ski comite", "ski comite holding", "other name", "ski college"]
    s = ["ski college", "ski college", "ski college", "ski college", "ski college", "lepicure club"]
    k = name_kinds(q, s, df, len(s1), frozenset({"holding"}))
    # order-free key; typo = rare swap; generic swap (both words common); filler ignored; other
    assert k.tolist() == [0, 1, 2, 2, 3, 3]


def test_anchor_detects_and_corrects_prior_of_one_kind():
    from ber.decide.anchor import correct, kind_ratios, pooled_bias
    rng = np.random.default_rng(1)
    h1 = np.array([0.07, 0.76, 0.04, 0.04, 0.03, 0.06])   # true copies: mostly equal number
    h0 = np.array([0.00, 0.10, 0.00, 0.05, 0.70, 0.15])   # decoys: mostly shifted

    def sample(n, pi_true, pi_model, kind):
        y = rng.random(n) < pi_true
        hn = np.where(y, rng.choice(6, n, p=h1), rng.choice(6, n, p=h0))
        p = pi_model * h1[hn] / (pi_model * h1[hn] + (1 - pi_model) * h0[hn])  # calibrated at pi_model
        return pd.DataFrame({"q_id": [f"{kind}{i}" for i in range(n)], "s1_id": "X", "p": p,
                             "hn_rel": hn.astype(float), "a_tsr": 100.0, "kind": kind, "y": y})

    # val: model calibrated in both kinds -> ratios ~1, bias ~1
    val = pd.concat([sample(40000, 0.85, 0.85, 0), sample(8000, 0.4, 0.4, 2)], ignore_index=True)
    rv = kind_ratios(val.kind.to_numpy(), val.hn_rel.to_numpy(), val.p.to_numpy())
    assert abs(rv["generic_swap"]["r"] - 1) < 0.25 and abs(rv["equal_key"]["r"] - 1) < 1e-6
    bias = pooled_bias({"A": rv})
    # test: generic swaps are 85 % true copies, the model still assumes 40 %
    test = pd.concat([sample(40000, 0.85, 0.85, 0), sample(8000, 0.85, 0.4, 2)], ignore_index=True)
    adj, rep = correct(test, test.kind.to_numpy(), bias, min_ratio=2.0, min_pairs=1000)
    g = rep["generic_swap"]
    assert abs(g["pi_anchor"] - 0.85) < 0.05 and g["r_applied"] > 2
    assert rep["equal_key"]["r_applied"] == 1.0
    m = (test.kind == 2).to_numpy()
    assert (adj.p.to_numpy()[~m] == test.p.to_numpy(np.float32)[~m]).all()
    # corrected p is better calibrated for the shifted kind
    assert abs(adj.p[m].mean() - test.y[m].mean()) < abs(test.p[m].mean() - test.y[m].mean())
    # a pair on another street is never touched
    far = test.assign(a_tsr=50.0)
    adj2, _ = correct(far, far.kind.to_numpy(), bias, min_ratio=2.0, min_pairs=1000)
    assert np.allclose(adj2.p, far.p.astype(np.float32))
