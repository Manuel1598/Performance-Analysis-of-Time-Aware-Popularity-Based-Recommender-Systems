"""Run one exact validation-first hyperparameter configuration.

This helper is intended for distributing a small number of remaining trials.
It writes to an isolated output directory and accepts only configurations that
belong to the protocol's declared grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tools.run_validation_first_experiments as experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["topn", "session"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = (
        experiments.TOPN_DATASETS
        if args.scenario == "topn"
        else experiments.SESSION_DATASETS
    )
    models = (
        experiments.TOPN_MODELS
        if args.scenario == "topn"
        else experiments.SESSION_MODELS
    )
    grid = experiments.topn_grid if args.scenario == "topn" else experiments.session_grid

    if args.dataset not in datasets:
        raise ValueError(f"{args.dataset!r} is not a {args.scenario} dataset")
    if args.model not in models:
        raise ValueError(f"{args.model!r} is not a {args.scenario} model")

    updates = json.loads(args.config_json)
    declared = {experiments.serialise_config(item) for item in grid(args.model)}
    serialised = experiments.serialise_config(updates)
    if serialised not in declared:
        raise ValueError(
            f"Configuration is not declared in the {args.model} {args.scenario} grid: "
            f"{serialised}"
        )

    experiments.configure_output_dir(args.output_dir)
    identifier = experiments.run_id(
        args.scenario, args.dataset, args.model, updates
    )
    if identifier in experiments.successful_ids(
        experiments.VALIDATION_FILE, "run_id"
    ):
        print(f"Skipping completed validation run: {identifier}", flush=True)
        return

    print(
        f"Validation run: {args.scenario}, {args.dataset}, "
        f"{args.model}, {updates}",
        flush=True,
    )
    result = experiments.run_validation_trial(
        args.scenario, args.dataset, args.model, updates, args.device
    )
    experiments.upsert_result(experiments.VALIDATION_FILE, result, "run_id")
    print(
        f"Finished: status={result['status']}, "
        f"valid_mrr@10={result.get('valid_mrr@10')}",
        flush=True,
    )
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
