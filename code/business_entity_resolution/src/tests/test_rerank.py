import numpy as np

from ber.rerank.stage import group_features


def test_group_features_query_and_s1_context():
    # queries 0,1 (source 2) and 2 (source 3); S1 10 and 11
    q = np.array([0, 0, 1, 2])
    s = np.array([10, 11, 10, 10])
    src = np.array([2, 2, 2, 3])
    p = np.array([0.9, 0.2, 0.6, 0.7])
    f = group_features(q, s, src, p)
    # query context
    assert list(f.q_rank) == [0, 1, 0, 0]
    np.testing.assert_allclose(f.q_best_other, [0.2, 0.9, -1, -1], atol=1e-6)
    np.testing.assert_allclose(f.q_margin, [0.7, -0.7, 0.6, 0.7], atol=1e-6)
    # S1 10 is listed by queries 0 (0.9), 1 (0.6), 2 (0.7)
    assert list(f.s_rank) == [0, 0, 2, 1]
    np.testing.assert_allclose(f.s_sum_other, [1.3, 0.0, 1.6, 1.5], atol=1e-6)
    np.testing.assert_allclose(f.s_n_hi_other, [2, 0, 2, 2])
    # same source (2) only: queries 0 and 1 compete on S1 10; query 2 is alone in source 3
    np.testing.assert_allclose(f.ss_best_other, [0.6, -1, 0.9, -1], atol=1e-6)
    # claims: every query's best S1 is 10 -> S1 10 has 3 claims, S1 11 none
    np.testing.assert_allclose(f.c_expected, [2.2, 0.0, 2.2, 2.2], atol=1e-6)
    assert f.dtypes.map(lambda d: d == np.float32).all()


def test_domain_normalize_per_group():
    from ber.rerank.stage import domain_normalize
    # columns: a count column, an untouched one, an IDF column, and one column beyond
    # the names (e.g. still empty)
    M = np.array([[2, 5, 0.0, 7],
                  [4, 5, 2.0, 7],
                  [10, 5, 4.0, 7],
                  [30, 5, 8.0, 7]], dtype=np.float32)
    cols = ["s_n", "p1", "k_idf_unmatched_q"]
    groups = np.array([0, 0, 1, 1])
    out = domain_normalize(M.copy(), cols, groups)
    np.testing.assert_allclose(out[:, 0], [2 / 3, 4 / 3, 0.5, 1.5], rtol=1e-6)   # / group mean
    np.testing.assert_allclose(out[:, 1], 5)                                     # not a count
    q0 = np.quantile([2.0], 0.99)
    q1 = np.quantile([4.0, 8.0], 0.99)
    np.testing.assert_allclose(out[:, 2], [0, 2 / q0, 4 / q1, 8 / q1], rtol=1e-6)  # / 99th pct of non-zero
    np.testing.assert_allclose(out[:, 3], 7)                                     # beyond cols


def test_loaded_s1_val_side_and_hold_regions():
    import pandas as pd
    from ber.rerank.stage import loaded_s1
    s1 = pd.DataFrame({"entity_id": ["a", "b", "c", "d"], "country": ["X", "X", "X", "Y"],
                       "geo": ["AA", "BB", "CC", "AA"], "side": ["val", "train", "train", "train"]})
    hold = pd.DataFrame({"country": ["X"], "geo": ["BB"], "n_s1": [1]})
    assert loaded_s1(s1, hold).to_dict() == {"a": True, "b": True, "c": False, "d": False}
    assert loaded_s1(s1, None).to_dict() == {"a": True, "b": False, "c": False, "d": False}


def test_domain_normalize_s1_counts_scaled_on_complete_groups_only():
    from ber.rerank.stage import domain_normalize
    # rows 0-2 complete, row 3 an artificially empty S1 group (training only)
    M = np.array([[4, 2], [6, 2], [8, 2], [0, 6]], dtype=np.float32)
    cols = ["c_expected", "q_n"]          # an S1-group count and a query-side count
    complete = np.array([True, True, True, False])
    out = domain_normalize(M.copy(), cols, None, complete)
    np.testing.assert_allclose(out[:, 0], [4 / 6, 1, 8 / 6, 0], rtol=1e-6)   # scale from rows 0-2
    np.testing.assert_allclose(out[:, 1], [2 / 3, 2 / 3, 2 / 3, 2], rtol=1e-6)  # all rows
    plain = domain_normalize(M.copy(), cols)
    np.testing.assert_allclose(plain[:, 0], [4 / 4.5, 6 / 4.5, 8 / 4.5, 0], rtol=1e-6)


def test_extra_features_context_and_alignment():
    import pandas as pd
    from ber.rerank.stage import _aligned_extra, extra_features
    q = np.array([0, 0, 1])
    s = np.array([5, 6, 5])
    ce = np.array([2.0, -1.0, 0.5])
    f = extra_features(q, s, ce)
    assert list(f["ce_q_rank"]) == [0, 1, 0]
    np.testing.assert_allclose(f["ce_q_margin"], [3.0, -3.0, 20.5])   # no other candidate: vs -20
    assert list(f["ce_s_rank"]) == [0, 0, 1]
    np.testing.assert_allclose(f["ce_s_margin"], [1.5, 19.0, -1.5])
    e = pd.DataFrame({"q_id": ["a", "a", "b"], "s1_id": ["x", "y", "x"], "ce": [2.0, -1.0, 0.5]})
    np.testing.assert_allclose(_aligned_extra([e], ["a", "a", "b"], ["x", "y", "x"]), [2.0, -1.0, 0.5])
    # another row order falls back to the join
    np.testing.assert_allclose(_aligned_extra([e], ["b", "a", "a"], ["x", "y", "x"]), [0.5, -1.0, 2.0])
