"""Smoke test: the package and its core dependencies import."""


def test_imports():
    import lightgbm  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import rapidfuzz  # noqa: F401
    import sklearn  # noqa: F401
    import yaml  # noqa: F401

    import ber

    assert ber.__version__
