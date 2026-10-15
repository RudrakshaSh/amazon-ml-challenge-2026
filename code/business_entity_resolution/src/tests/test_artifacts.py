import json

import pandas as pd
import pytest

from ber.artifacts import StageError, StageRun
from ber.cli import COMMANDS, build_parser, main


def test_stage_writes_manifest_and_outputs(cfg):
    with StageRun(cfg, "a") as st:
        st.write_parquet(pd.DataFrame({"x": [1, 2, 3]}), "table")
        st.metric("m", 0.5)
    m = json.loads((cfg.run_dir / "a" / "manifest.json").read_text())
    assert m["status"] == "ok" and m["metrics"] == {"m": 0.5}
    assert m["outputs"]["table"]["rows"] == 3 and m["config_hash"] == cfg.hash()
    assert (cfg.run_dir / "a" / "log.txt").read_text()
    assert (cfg.run_dir / "config.yaml").exists()

    with StageRun(cfg, "b") as st:
        df = st.read_input("a", "table")
        assert df.x.tolist() == [1, 2, 3]
    m = json.loads((cfg.run_dir / "b" / "manifest.json").read_text())
    assert m["inputs"]["test:a/table"]["config_hash"] == cfg.hash()


def test_failed_stage_is_recorded_and_blocks_downstream(cfg):
    with pytest.raises(ZeroDivisionError):
        with StageRun(cfg, "bad"):
            1 / 0
    m = json.loads((cfg.run_dir / "bad" / "manifest.json").read_text())
    assert m["status"] == "failed" and "ZeroDivisionError" in m["error"]
    with StageRun(cfg, "next") as st, pytest.raises(StageError, match="status 'failed'"):
        st.input_path("bad", "x")


def test_missing_upstream_has_actionable_error(cfg):
    with StageRun(cfg, "next") as st, pytest.raises(StageError, match="ber missing"):
        st.input_path("missing", "x")


def test_cli_lists_all_stages(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


def test_cli_noop_writes_manifest(tmp_path):
    (tmp_path / "dataset").mkdir()
    assert main(["noop", "--set", f"paths.root={tmp_path}", "--run", "t"]) == 0
    m = json.loads((tmp_path / "artifacts" / "t" / "noop" / "manifest.json").read_text())
    assert m["status"] == "ok" and m["metrics"]["answer"] == 42


def test_cli_config_error_exit_code(capsys):
    assert main(["config", "--set", "bogus=1"]) == 2
    assert "unknown config key" in capsys.readouterr().err


def test_all_stages_implemented():
    from ber.cli import COMMANDS
    stages = ["split", "normalize", "candidates", "features", "features-hold", "scorer",
              "crossenc", "rerank", "decide", "submit"]
    assert all(s in COMMANDS and callable(COMMANDS[s].func) for s in stages)
