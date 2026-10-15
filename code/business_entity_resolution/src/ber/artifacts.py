"""Stage runs: one directory, log file and manifest per stage.

    with StageRun(cfg, "candidates") as st:
        records = st.read_input("normalize", "records")
        ...
        st.write_parquet(cands, "candidates")
        st.metric("pair_recall", 0.998)

produces ``artifacts/<run>/candidates/`` with ``candidates.parquet``,
``log.txt`` and ``manifest.json``. The manifest records status (ok/failed plus
traceback), timings, peak memory, config hash, git commit, inputs (with their
upstream config hashes), outputs (rows/bytes) and metrics. It is written even
when the stage fails, so every run can be inspected afterwards.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import __version__
from .config import Config
from .io import read_parquet, write_parquet
from .log import (add_file_handler, get_logger, peak_rss_gb, remove_handler,
                  setup_logging)

MANIFEST = "manifest.json"


class StageError(RuntimeError):
    pass


def git_state(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "--short", "HEAD") or None,
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def load_manifest(cfg: Config, stage: str) -> dict[str, Any]:
    path = cfg.run_dir / stage / MANIFEST
    if not path.exists():
        raise StageError(f"no '{stage}' output in run '{cfg.run_id}' ({path}); "
                         f"run `ber {stage} --run {cfg.run_id}` first")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("status") != "ok":
        raise StageError(f"stage '{stage}' in run '{cfg.run_id}' has status "
                         f"{m.get('status')!r}; see {path.parent / 'log.txt'}")
    return m


class StageRun:
    def __init__(self, cfg: Config, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.dir = cfg.run_dir / stage
        self.log = get_logger(stage)
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

    # -------------------------------------------------------- lifecycle
    def __enter__(self) -> StageRun:
        setup_logging(self.cfg.log_level)  # stages log even when not started via the CLI
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.data.phase == "test" and (self.dir / MANIFEST).exists():
            # test-only pass: keep the train-side outputs/metrics of the earlier run
            prev = load_manifest(self.cfg, self.stage)
            self.outputs.update(prev.get("outputs", {}))
            self.metrics.update(prev.get("metrics", {}))
        self._handler = add_file_handler(self.dir / "log.txt")
        self._t0 = time.perf_counter()
        self._started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(self.cfg.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg.to_dict(), f, sort_keys=False)
        self.log.info("=== stage %s | run %s | config %s | %s", self.stage,
                      self.cfg.run_id, self.cfg.hash(), self.dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = time.perf_counter() - self._t0
        manifest = {
            "stage": self.stage,
            "run_id": self.cfg.run_id,
            "status": "ok" if exc is None else "failed",
            "error": None if exc is None else "".join(
                traceback.format_exception(exc_type, exc, tb)),
            "started_utc": self._started,
            "duration_s": round(duration, 2),
            "peak_rss_gb": round(peak_rss_gb(), 3),
            "config_hash": self.cfg.hash(),
            "git": git_state(self.cfg.root),
            "ber_version": __version__,
            "python": platform.python_version(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
        }
        with open(self.dir / MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        if exc is None:
            self.log.info("=== stage %s ok in %.1fs (peak RSS %.2f GB)", self.stage,
                          duration, peak_rss_gb())
        else:
            self.log.error("=== stage %s FAILED after %.1fs: %s", self.stage, duration, exc)
        remove_handler(self._handler)
        return False  # never swallow exceptions

    # ------------------------------------------------------------ inputs
    def input_path(self, stage: str, name: str, run: str | None = None) -> Path:
        """Path of output ``name`` of an earlier ``stage`` in this run (or in ``run``)."""
        cfg = self.cfg if run in (None, self.cfg.run_id) else replace(self.cfg, run_id=run)
        m = load_manifest(cfg, stage)
        if name not in m["outputs"]:
            raise StageError(f"stage '{stage}' has no output '{name}'; "
                             f"available: {sorted(m['outputs'])}")
        path = cfg.run_dir / stage / m["outputs"][name]["file"]
        self.inputs[f"{cfg.run_id}:{stage}/{name}"] = {"file": str(path),
                                                       "config_hash": m["config_hash"]}
        if run in (None, self.cfg.run_id) and m["config_hash"] != self.cfg.hash():
            self.log.warning("input %s/%s was built with config %s (current %s)",
                             stage, name, m["config_hash"], self.cfg.hash())
        return path

    def read_input(self, stage: str, name: str, columns: list[str] | None = None,
                   run: str | None = None) -> pd.DataFrame:
        return read_parquet(self.input_path(stage, name, run), columns)

    # ----------------------------------------------------------- outputs
    def path(self, filename: str) -> Path:
        return self.dir / filename

    def _record(self, name: str, path: Path, **extra: Any) -> None:
        size = (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir() else path.stat().st_size)
        self.outputs[name] = {"file": path.name, "bytes": size, **extra}

    def write_parquet(self, df: pd.DataFrame, name: str) -> Path:
        path = self.path(f"{name}.parquet")
        write_parquet(df, path)
        self._record(name, path, rows=len(df), columns=list(df.columns))
        self.log.info("wrote %s: %d rows x %d cols", path.name, len(df), df.shape[1])
        return path

    def write_json(self, obj: Any, name: str) -> Path:
        path = self.path(f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        self._record(name, path)
        return path

    def register(self, name: str, path: Path, **extra: Any) -> None:
        """Record a file or directory written some other way (a TSV, a directory of
        parquet parts, ...) as an output."""
        self._record(name, path, **extra)

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        self.log.info("metric %s = %s", key, value)
