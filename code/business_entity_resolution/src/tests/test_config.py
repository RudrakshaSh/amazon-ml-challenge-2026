import pytest

from ber.config import ConfigError, DEFAULT_CONFIG, CONFIG_DIR, load_config


def test_default_loads():
    cfg = load_config()
    assert cfg.run_id == "main"
    assert cfg.candidates.k_name == 10
    assert (cfg.root / "dataset").is_dir()


def test_dev_extends_default():
    cfg = load_config([CONFIG_DIR / "dev.yaml"])
    assert cfg.run_id == "dev" and cfg.data.slice == "dev"
    assert cfg.candidates.ngram == 3  # inherited


def test_overrides_are_yaml_typed():
    cfg = load_config(overrides=["candidates.k_name=25", "data.slice=null"], run_id="x")
    assert cfg.candidates.k_name == 25 and cfg.data.slice is None and cfg.run_id == "x"


@pytest.mark.parametrize("expr, msg", [
    ("candidates.k_nmae=3", "unknown config key"),
    ("candidates.k_name=ten", "expected"),
    ("candidates=5", "expected a section"),
    ("seed=null", "may not be null"),
    ("nokey", "key.sub=value"),
])
def test_bad_config_is_rejected(expr, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(overrides=[expr])


def test_circular_extends(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config([tmp_path / "a.yaml"])


def test_hash_ignores_run_id_but_not_params():
    a = load_config(run_id="a")
    assert a.hash() == load_config(run_id="b").hash()
    assert a.hash() != load_config(overrides=["seed=1"]).hash()
    assert DEFAULT_CONFIG.exists()


def test_list_and_optional_fields():
    cfg = load_config(overrides=["decide.sweep=[0.1, 0.5]", "decide.threshold=0.3"])
    assert cfg.decide.sweep == [0.1, 0.5] and cfg.decide.threshold == 0.3
    assert load_config().decide.threshold is None
    with pytest.raises(ConfigError, match="expected"):
        load_config(overrides=["decide.sweep=0.5"])
