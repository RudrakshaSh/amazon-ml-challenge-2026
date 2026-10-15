import numpy as np
import pandas as pd
import pytest

from ber.config import load_config
from ber.eval.split import choose_regions, make_split, slice_mask, stable_unit


def test_choose_regions_hits_target_without_big_overshoot():
    sizes = pd.Series({f"r{i}": 10 for i in range(100)})
    chosen = choose_regions(sizes, 100, np.random.default_rng(0))
    assert 95 <= sizes[chosen].sum() <= 115


def test_choose_regions_skips_oversized_region():
    sizes = pd.Series({"huge": 1000, "a": 30, "b": 30, "c": 30})
    chosen = choose_regions(sizes, 100, np.random.default_rng(0))
    assert "huge" not in chosen


def test_stable_unit_is_order_independent():
    ids = pd.Series([f"S2-{i}" for i in range(1000)])
    u1 = stable_unit(ids, 1)
    u2 = stable_unit(ids[::-1].reset_index(drop=True), 1)[::-1]
    assert np.allclose(u1, u2) and 0 <= u1.min() and u1.max() < 1
    assert not np.allclose(u1, stable_unit(ids, 2))


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    regions = [f"R{i}" for i in range(40)]
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(4000)],
                       "country": "US", "geo": rng.choice(regions, 4000)})
    s1.loc[:9, "geo"] = ""  # a few with no detectable region
    owners = s1.entity_id.sample(3000, random_state=0).to_numpy()
    q = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(4000)], "country": "US",
                      "geo": rng.choice(regions + [""], 4000),
                      "owner": list(owners) + [""] * 1000})
    return s1, q


def test_make_split_invariants(toy):
    s1, q = toy
    cfg = load_config(overrides=["split.val_frac=0.2", "split.dev_train_frac=0.1",
                                 "split.dev_val_frac=0.05"])
    s, qs, regions = make_split(s1, q, cfg)
    assert s.side.isin(["train", "val"]).all() and qs.side.isin(["train", "val"]).all()
    assert 0.15 < (s.side == "val").mean() < 0.25
    # matched queries follow their owner
    m = qs[qs.owner != ""]
    own = s.set_index("entity_id").loc[m.owner]
    assert (m.side.to_numpy() == own.side.to_numpy()).all()
    assert (m.dev.to_numpy() == own.dev.to_numpy()).all()
    # unmatched with a known region follow that region
    lut = regions.set_index("geo").side
    um = qs[(qs.owner == "") & (qs.geo != "")]
    assert (um.side.to_numpy() == lut.loc[um.geo].to_numpy()).all()
    assert set(qs.basis) == {"owner", "own_region", "hash"}
    # a region is entirely on one side
    assert (s[s.geo != ""].groupby("geo").side.nunique() == 1).all()
    assert s.dev.any() and (s.dev & (s.side == "val")).any()


def test_dev_slice_has_val_even_when_val_regions_are_big():
    # few huge regions + some small ones: the dev-val target (1%) is far below
    # any big region's size, so dev-val must come from the small regions
    geo = ["BIG1"] * 3000 + ["BIG2"] * 3000 + ["BIG3"] * 3000 + \
          [f"s{i}" for i in range(40) for _ in range(50)]
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(len(geo))],
                       "country": "US", "geo": geo})
    q = pd.DataFrame(columns=["entity_id", "country", "geo", "owner"])
    s, _, _ = make_split(s1, q, load_config())
    assert (s.dev & (s.side == "val")).sum() > 0
    assert (s.dev & (s.side == "train")).sum() > 0
    assert 0.07 < (s.side == "val").mean() < 0.13


def test_make_split_is_deterministic(toy):
    s1, q = toy
    cfg = load_config()
    a = make_split(s1, q, cfg)
    b = make_split(s1, q, cfg)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_slice_mask():
    df = pd.DataFrame({"side": ["train", "val", "val"], "dev": [True, False, True]})
    assert slice_mask(df, load_config()).all()
    assert slice_mask(df, load_config(overrides=["data.slice=dev"])).tolist() == [True, False, True]
    assert slice_mask(df, load_config(overrides=["data.slice=dev"]), "val").tolist() == [False, False, True]
    with pytest.raises(ValueError, match="unknown data.slice"):
        slice_mask(df, load_config(overrides=["data.slice=tiny"]))


def test_choose_regions_falls_back_to_closest_region():
    # real India S1 sizes: three tiny regions, then nothing below ~13k; dev-val
    # target is ~8.8k, so the pick must not end up as just the tiny regions
    sizes = pd.Series({"UK": 1, "CH": 3, "MN": 4, "PB": 13014, "OD": 13898, "AP": 17377,
                       "MP": 22503, "DL": 121337, "MH": 191679})
    chosen = choose_regions(sizes, 8832, np.random.default_rng(0))
    assert sizes[chosen].sum() > 4400 and "PB" in chosen


def test_hold_queries_follow_their_owner():
    from ber.eval.split import hold_queries
    s1 = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "geo": ["AA", "BB", "AA"]})
    q = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S3-5", "S3-6"],
        "country": ["X"] * 6,
        # 1: copy of a (hold) with no region; 2: copy of a written in BB; 3: copy of b
        # written in AA; 4: unmatched in AA; 5: unmatched in BB; 6: val side
        "geo": ["", "BB", "AA", "AA", "BB", "AA"],
        "owner": ["S1-a", "S1-a", "S1-b", "", "", "S1-c"],
        "side": ["train"] * 5 + ["val"]})
    hold = pd.DataFrame({"country": ["X"], "geo": ["AA"], "n_s1": [2]})
    by_owner = hold_queries(q, s1, hold)
    assert sorted(by_owner.entity_id) == ["S2-1", "S2-2", "S2-4"]
    assert set(by_owner.geo) == {"AA"}
