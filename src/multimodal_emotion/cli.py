"""Command-line preflight utilities. No command starts training implicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, config_hash, load_config
from .queue import validate_run_queue


DEFAULT_QUEUE = Path(__file__).resolve().parents[2] / "configs" / "core_experiment_queue.csv"


def preflight(config_path: Path, queue_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    rows = validate_run_queue(queue_path, config)
    return {
        "status": "PASS",
        "config_hash": config_hash(config),
        "planned_core_runs": len(rows),
        "models": config["training"]["model_execution_order"],
        "seeds": config["training"]["seed_execution_order"],
        "validation_views_per_run": config["evaluation"]["total_views_per_run"],
        "data_paths_filled": all(config["paths_to_fill_at_E00"].values()),
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="multimodal-emotion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight", help="check config and planned run queue only")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()

    if args.command == "preflight":
        print(json.dumps(preflight(args.config, args.queue), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
