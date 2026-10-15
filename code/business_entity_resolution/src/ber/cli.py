"""Command line entry point: ``ber <stage> [--config ...] [--set k=v] [--run ID]``.

Each pipeline stage is one sub-command, which reads the previous stage's
artifacts from ``artifacts/<run>/`` and writes its own.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import yaml

from .artifacts import StageRun
from .config import Config, ConfigError, load_config
from .log import get_logger, setup_logging

log = get_logger("cli")


@dataclass(frozen=True)
class Command:
    func: Callable[[Config, argparse.Namespace], None]
    help: str


def cmd_config(cfg: Config, args: argparse.Namespace) -> None:
    """Print the fully resolved config (after extends and overrides)."""
    print(f"# run_id={cfg.run_id} config_hash={cfg.hash()} root={cfg.root}")
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False), end="")


def cmd_noop(cfg: Config, args: argparse.Namespace) -> None:
    """Exercise the stage machinery: writes a manifest, a log and a small artifact."""
    with StageRun(cfg, "noop") as st:
        st.write_json({"hello": "world"}, "result")
        st.metric("answer", 42)


def cmd_split(cfg: Config, args: argparse.Namespace) -> None:
    from .eval.split import run_split
    run_split(cfg)


def cmd_normalize(cfg: Config, args: argparse.Namespace) -> None:
    from .normalize.stage import run_normalize
    run_normalize(cfg)


def cmd_candidates(cfg: Config, args: argparse.Namespace) -> None:
    from .candidates.generate import run_candidates
    run_candidates(cfg)


def cmd_features(cfg: Config, args: argparse.Namespace) -> None:
    from .features.stage import run_features
    run_features(cfg)


def cmd_features_hold(cfg: Config, args: argparse.Namespace) -> None:
    from .features.stage import run_features_hold
    run_features_hold(cfg)


def cmd_scorer(cfg: Config, args: argparse.Namespace) -> None:
    from .scorer.gbm import run_scorer
    run_scorer(cfg)


def cmd_crossenc(cfg: Config, args: argparse.Namespace) -> None:
    from .crossenc.stage import run_crossenc
    run_crossenc(cfg)


def cmd_rerank(cfg: Config, args: argparse.Namespace) -> None:
    from .rerank.stage import run_rerank
    run_rerank(cfg)


def cmd_decide(cfg: Config, args: argparse.Namespace) -> None:
    from .decide.stage import run_decide
    run_decide(cfg)


def cmd_submit(cfg: Config, args: argparse.Namespace) -> None:
    from .submit import run_submit
    run_submit(cfg)


def cmd_explain(cfg: Config, args: argparse.Namespace) -> None:
    from .explain import explain
    explain(cfg, args.entity_id)


COMMANDS: dict[str, Command] = {
    "config": Command(cmd_config, "print the resolved config"),
    "noop": Command(cmd_noop, "no-op stage (tests the artifact/manifest machinery)"),
    "split": Command(cmd_split, "geographic train/val split + dev slice"),
    "normalize": Command(cmd_normalize, "stage 0: normalise train + test records"),
    "candidates": Command(cmd_candidates, "stage 1: TF-IDF top-k candidates + recall"),
    "features": Command(cmd_features, "stage 2: pair + context features"),
    "features-hold": Command(cmd_features_hold,
                             "features of the held-out train regions (rerank.hold_frac)"),
    "scorer": Command(cmd_scorer, "stage 2: train/apply LightGBM scorer"),
    "crossenc": Command(cmd_crossenc, "transformer cross-encoder: fine-tune, score stage-2 pairs"),
    "rerank": Command(cmd_rerank, "stage 2b: re-rank with S1/query group context (cross-fitted)"),
    "decide": Command(cmd_decide, "stage 4: one-owner assignment + expected-F0.5, val F0.5"),
    "submit": Command(cmd_submit, "write output/*.tsv for test and run the validator"),
    "explain": Command(cmd_explain, "trace one entity through every stage"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ber", description="Business entity resolution pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="<stage>")
    for name, c in COMMANDS.items():
        sp = sub.add_parser(name, help=c.help, description=c.help)
        sp.add_argument("--config", "-c", action="append", default=None,
                        help="YAML config file(s), merged in order "
                             "(default: ber/configs/default.yaml)")
        sp.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set candidates.k_name=20")
        sp.add_argument("--run", "-r", default=None, help="run id (artifacts/<run>/)")
        if name == "explain":
            sp.add_argument("entity_id", help="S1-/S2-/S3- id to trace")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.set, args.run)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.log_level)
    COMMANDS[args.command].func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
