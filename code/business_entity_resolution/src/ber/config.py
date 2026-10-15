"""Typed pipeline configuration loaded from YAML.

* Files may ``extends: other.yaml`` (resolved relative to the file); later
  files and ``--set a.b=value`` overrides are deep-merged on top.
* Loading is strict: unknown keys and wrong scalar types raise
  :class:`ConfigError`, so a typo can't silently fall back to a default.
* Paths in ``paths`` are relative to the project root (the ``student_resource``
  directory, which contains ``dataset/``), unless absolute.
"""

from __future__ import annotations

import hashlib
import json
import os
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


class ConfigError(ValueError):
    pass


PHASES = ("all", "train", "test")


# ---------------------------------------------------------------- schema
# Stage sections only hold parameters that exist today; tasks add fields as
# they introduce them (and document them in configs/default.yaml).


@dataclass
class PathsConfig:
    root: str = ""  # "" = auto-detect (BER_ROOT env, or nearest dir containing dataset/)
    data_dir: str = "dataset"
    artifacts_dir: str = "artifacts"
    output_dir: str = "output"


@dataclass
class DataConfig:
    slice: str | None = None  # None = full data; "dev" = dev slice (T0.3)
    split_run: str = "main"   # run whose `split` stage defines sides and the dev slice
    with_test: bool = True    # also process the test split (needs full data, no slice)
    phase: str = "all"        # "all" | "train" (train/val side only) | "test" (test side only,
                              # reusing the train-side outputs already in the stage dirs)


@dataclass
class NormalizeConfig:
    workers: int = 0              # processes; 0 = cpu_count - 2
    chunk_size: int = 100_000     # records per worker task
    dict_min_count: int = 2       # translit dictionary: min aligned occurrences
    dict_min_share: float = 0.6   # translit dictionary: min share of the dominant spelling


@dataclass
class SplitConfig:
    val_frac: float = 0.10        # share of each country's S1 held out (whole regions)
    dev_train_frac: float = 0.02  # dev slice: share of the country taken from train regions
    dev_val_frac: float = 0.01    # dev slice: share of the country taken from val regions


@dataclass
class CandidatesConfig:
    ngram: int = 3
    k_name: int = 10
    k_addr: int = 10
    k_combo: int = 10
    max_df: float = 0.05          # drop n-grams in more than this share of S1 (speed)
    min_sim: float = 0.0          # drop top-k hits below this cosine
    chunk_size: int = 20_000
    n_threads: int = 0            # 0 = cpu_count - 2
    block_by_region: bool = True
    region_min_share: float = 0.001  # neighbourhood: owner regions with >= this share
    k_name_noaddr: int = 30       # name candidates for queries with an empty address
    device: str = "auto"          # top-k search: "cpu" | "gpu" (CuPy) | "auto"
    gpu_mem_gb: float = 1.5       # GPU memory for one chunk's dense similarity block
    gpu_min_s1: int = 50_000      # blocks with fewer S1 records search on the CPU


@dataclass
class FeaturesConfig:
    train_query_frac: float = 0.2  # share of train-side queries used (val side: all)
    chunk_rows: int = 1_000_000    # candidate pairs per feature chunk
    name_key: bool = True          # v2: filler-free name keys + IDF token overlap
    filler_min_rate: float = 0.0005  # filler words: min share of S2/S3 names ...
    filler_min_ratio: float = 2.0    # ... and min ratio to their share of S1 names


@dataclass
class ScorerConfig:
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    num_boost_round: int = 3000
    early_stopping: int = 100
    n_threads: int = 0
    train_query_frac: float = 0.14  # share of featurised train-side queries used for training
    es_query_frac: float = 0.05     # share of val-side queries used for early stopping
    drop_features: list[str] = field(default_factory=list)  # ablations: names or "prefix*"


@dataclass
class RerankConfig:
    min_p1: float = 0.001           # drop pairs below this stage-1 p before stage 2
    folds: int = 4                  # cross-fitting folds on the val side
    fold_by: str = "s1"             # "s1" (groups by the record's top S1) | "region" (check)
    domain_norm: bool = True        # size-dependent features / their per-country-split scale
    hold_frac: float = 0.4          # v2.4: share of each country's S1 in extra held-out train regions
                                    # (ber features-hold; stage 1 trains without them)
    train_sides: list[str] = field(default_factory=lambda: ["val", "hold"])  # stage-2 training pairs:
                                    # val and/or hold; a side not trained on is scored by the fold mean
    train_complete: bool = True     # v2.5: train only on pairs whose S1 group is complete (S1 in a val
                                    # or hold region); the others are still scored
    norm_complete: bool = True      # domain_norm: S1-group counts take their training scale from
                                    # complete-group pairs only (as every test group is complete)
    extra_scores: str | None = "crossenc"  # "crossenc" (that stage's scores) or a dir with val/hold/test_<country>.parquet
                                    # (q_id, s1_id, ce) in stage 2's row order: an external pair score
                                    # (cross-encoder logit) + its query / S1-group context as features
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    lambda_l2: float = 1.0
    feature_fraction: float = 0.9
    num_boost_round: int = 2000
    early_stopping: int = 50
    raw_features: list[str] = field(default_factory=lambda: ["all"])  # or a feature list


@dataclass
class CrossencConfig:
    """``ber crossenc``: a transformer cross-encoder whose pair score
    feeds stage 2 (``rerank.extra_scores: crossenc``)."""
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Apache-2.0, 22 M parameters
    max_len: int = 64               # tokens per pair (record text + S1 text)
    batch_size: int = 256
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup: float = 0.05            # share of the steps with a linear warm-up
    epochs: int = 2
    infer_batch: int = 2048         # halved automatically on CUDA out-of-memory
    chunk: int = 1_000_000          # pairs tokenised at a time


@dataclass
class DecideConfig:
    method: str = "expected_f"      # "threshold" | "expected_f"
    threshold: float | None = None  # threshold method: None = best on val sweep
    min_p: float | None = None      # expected_f: drop p below this; None = best on val sweep
    empty_prior: float = 1.0        # expected_f: multiplier on P(no true match)
    tie_duplicates: bool = True     # exact duplicate S2/S3 records share one owner
    prior_shift: str = "anchor"     # "none" | "anchor" (per name-change kind, from the hn_rel mixture)
    anchor_min_ratio: float = 2.0   # anchor: correct a kind only if its de-biased odds ratio is beyond this
    anchor_min_pairs: int = 2000    # anchor: ... and it has at least this many same-street pairs
    anchor_max_ratio: float = 6.0   # anchor: cap on the applied odds ratio
    scores_stage: str = "rerank"    # "scorer" (stage 1 only) or "rerank" (stage 2)
    sweep: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                                        0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8])


@dataclass
class SubmitConfig:
    write_candidates: bool = True   # output/candidate_pairs.tsv = input pairs of the final model


@dataclass
class Config:
    run_id: str = "main"
    seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    crossenc: CrossencConfig = field(default_factory=CrossencConfig)
    decide: DecideConfig = field(default_factory=DecideConfig)
    submit: SubmitConfig = field(default_factory=SubmitConfig)

    # ------------------------------------------------------ derived paths
    @property
    def root(self) -> Path:
        return find_root(self.paths.root)

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.root / path

    @property
    def run_test(self) -> bool:
        """Whether stages should also process the test split."""
        return (self.data.slice is None and self.data.with_test
                and self.data.phase in ("all", "test"))

    @property
    def run_train(self) -> bool:
        """Whether stages should process the train split (train + val sides).
        False only for ``data.phase=test``: the test pass reuses the model,
        region neighbourhoods and decision parameters of an earlier train pass."""
        if self.data.phase not in PHASES:
            raise ConfigError(f"unknown data.phase {self.data.phase!r}; valid: {PHASES}")
        return self.data.phase in ("all", "train")

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir) / self.run_id

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    # ------------------------------------------------------- serialising
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Stable short hash of the full config (excluding run_id and data.phase,
        which only choose which part of a stage runs)."""
        d = self.to_dict()
        d.pop("run_id")
        d["data"].pop("phase")
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------- loading


def find_root(explicit: str = "") -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("BER_ROOT"):
        return Path(os.environ["BER_ROOT"]).expanduser().resolve()
    for start in (Path.cwd(), PACKAGE_DIR):
        for d in (start, *start.parents):
            if (d / "dataset").is_dir():
                return d
    raise ConfigError("project root not found: set paths.root or BER_ROOT "
                      "(the directory that contains dataset/)")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular extends: {' -> '.join(map(str, (*_seen, path)))}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = d.pop("extends", None)
    if parent:
        d = _deep_merge(_load_yaml(path.parent / parent, (*_seen, path)), d)
    return d


def _apply_override(d: dict, expr: str) -> None:
    if "=" not in expr:
        raise ConfigError(f"override must look like key.sub=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {expr!r}: {p} is not a section")
    node[parts[-1]] = yaml.safe_load(raw)


def _check_scalar(hint: Any, value: Any, where: str) -> None:
    # list[float] / dict[...] are containers, not unions: don't unpack their args
    if typing.get_origin(hint) in (list, dict):
        allowed: tuple = (hint,)
    else:
        allowed = typing.get_args(hint) or (hint,)
    if value is None:
        if type(None) not in allowed:
            raise ConfigError(f"{where}: may not be null")
        return
    for t in allowed:
        if t is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            return
        if t in (int, str, bool) and type(value) is t:
            return
        if typing.get_origin(t) in (list, dict) and isinstance(value, typing.get_origin(t)):
            return
        if t in (list, dict) and isinstance(value, t):
            return
    raise ConfigError(f"{where}: expected {hint}, got {type(value).__name__} ({value!r})")


def from_dict(cls: type, d: dict, where: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"unknown config key(s) {sorted(where + k for k in unknown)}; "
                          f"valid under '{where or '<top>'}': {sorted(names)}")
    kwargs = {}
    for k, v in d.items():
        hint = hints[k]
        if is_dataclass(hint):
            if not isinstance(v, dict):
                raise ConfigError(f"{where}{k}: expected a section")
            kwargs[k] = from_dict(hint, v, f"{where}{k}.")
        else:
            _check_scalar(hint, v, f"{where}{k}")
            kwargs[k] = v
    return cls(**kwargs)


def load_config(paths: list[str | Path] | None = None,
                overrides: list[str] | None = None,
                run_id: str | None = None) -> Config:
    """Merge YAML files (default: configs/default.yaml), then overrides."""
    d: dict = {}
    for p in paths or [DEFAULT_CONFIG]:
        d = _deep_merge(d, _load_yaml(Path(p)))
    for expr in overrides or []:
        _apply_override(d, expr)
    if run_id:
        d["run_id"] = run_id
    return from_dict(Config, d)
