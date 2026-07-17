"""Compact, resumable tuning for the audited VSKNN implementation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNN
from src.recbole_framework.runners.session.run_vsknn_recbole import (
    SAMPLE_DATASETS,
    build_config,
)


BASELINE_CONFIG = {
    "neighbor_size": 100,
    "sample_size": 500,
    "sampling": "recent",
    "similarity": "vec",
    "session_weighting": "div",
    "score_weighting": "div",
}


def tuning_configs() -> list[dict]:
    """Return a compact one-factor-at-a-time grid around the baseline."""
    variants = [
        {},
        {"neighbor_size": 200},
        {"sample_size": 250},
        {"sample_size": 1000},
        {"similarity": "cosine"},
        {"session_weighting": "same"},
        {"session_weighting": "quadratic"},
        {"score_weighting": "same"},
        {"score_weighting": "quadratic"},
    ]
    return [{**BASELINE_CONFIG, **variant} for variant in variants]


def make_run_id(dataset: str, config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}::{digest}"


def select_best(results: pd.DataFrame) -> pd.DataFrame:
    successful = results[results["status"] == "success"].copy()
    if successful.empty:
        return successful
    successful["mrr@10"] = pd.to_numeric(successful["mrr@10"], errors="coerce")
    best_indices = successful.groupby("dataset")["mrr@10"].idxmax()
    return successful.loc[best_indices].sort_values("dataset").reset_index(drop=True)


def load_results(output_file: Path) -> pd.DataFrame:
    if not output_file.exists():
        return pd.DataFrame()
    return pd.read_csv(output_file)


def save_result(output_file: Path, result: dict) -> None:
    existing = load_results(output_file)
    updated = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
    updated.to_csv(output_file, index=False)


def seed_validated_baselines(project_root: Path, output_file: Path) -> None:
    if output_file.exists():
        return
    summary_file = (
        project_root
        / "recbole_results"
        / "vsknn_audited"
        / "all_session_samples_summary.csv"
    )
    if not summary_file.exists():
        return

    baseline_rows = pd.read_csv(summary_file)
    seeded = []
    for row in baseline_rows.to_dict("records"):
        dataset = row["dataset"]
        seeded.append(
            {
                **row,
                "run_id": make_run_id(dataset, BASELINE_CONFIG),
                "status": "success",
                "source": "validated_baseline",
                "error": "",
                "logged_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    pd.DataFrame(seeded).to_csv(output_file, index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SAMPLE_DATASETS,
        default=list(SAMPLE_DATASETS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=None,
        help="Stop after this many new runs; useful for smoke tests.",
    )
    return parser.parse_args(argv)


def tune_dataset(
    project_root: Path,
    dataset_name: str,
    device: str,
    output_file: Path,
    completed_run_ids: set[str],
    remaining_run_budget: int | None,
) -> int:
    configs = tuning_configs()
    pending = [
        config
        for config in configs
        if make_run_id(dataset_name, config) not in completed_run_ids
    ]
    if remaining_run_budget is not None:
        pending = pending[:remaining_run_budget]
    if not pending:
        print(f"{dataset_name}: no pending configurations")
        return 0

    data_config = build_config(
        project_root,
        dataset_name,
        BASELINE_CONFIG["neighbor_size"],
        BASELINE_CONFIG["sample_size"],
        device,
        False,
    )
    dataset = create_dataset(data_config)
    train_data, valid_data, test_data = data_preparation(data_config, dataset)

    completed = 0
    for position, config_values in enumerate(pending, start=1):
        run_id = make_run_id(dataset_name, config_values)
        print(
            f"\n{dataset_name}: run {position}/{len(pending)} "
            f"{json.dumps(config_values, sort_keys=True)}"
        )
        started_at = time.perf_counter()
        try:
            config = build_config(
                project_root,
                dataset_name,
                config_values["neighbor_size"],
                config_values["sample_size"],
                device,
                False,
            )
            config.final_config_dict.update(config_values)
            model = VSKNN(config, train_data.dataset).to(config["device"])
            trainer = Trainer(config, model)
            trainer.fit(train_data, valid_data, saved=False, verbose=False)
            metrics = dict(trainer.evaluate(test_data, load_best_model=False))
            status = "success"
            error = ""
        except Exception as exc:  # preserve progress and continue the grid
            metrics = {}
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "run_id": run_id,
            "model": "VSKNN-audited",
            "dataset": dataset_name,
            "seed": 42,
            "device": device,
            **config_values,
            "runtime_seconds": round(time.perf_counter() - started_at, 2),
            **metrics,
            "status": status,
            "source": "compact_tuning",
            "error": error,
            "logged_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_result(output_file, result)
        completed_run_ids.add(run_id)
        completed += 1
        print(f"status={status}, mrr@10={metrics.get('mrr@10')}")
    return completed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "recbole_results" / "vsknn_audited"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "compact_tuning_results.csv"
    best_file = output_dir / "compact_tuning_best_by_dataset.csv"

    seed_validated_baselines(project_root, output_file)
    existing = load_results(output_file)
    completed_run_ids = (
        set(existing.loc[existing["status"] == "success", "run_id"])
        if not existing.empty
        else set()
    )

    new_runs = 0
    for dataset_name in args.datasets:
        remaining = (
            None
            if args.max_new_runs is None
            else max(args.max_new_runs - new_runs, 0)
        )
        if remaining == 0:
            break
        new_runs += tune_dataset(
            project_root,
            dataset_name,
            args.device,
            output_file,
            completed_run_ids,
            remaining,
        )

    results = load_results(output_file)
    best = select_best(results)
    best.to_csv(best_file, index=False)
    print(f"\nNew runs: {new_runs}")
    print("Results:", output_file)
    print("Best by dataset:", best_file)
    if not best.empty:
        columns = [
            "dataset",
            "neighbor_size",
            "sample_size",
            "similarity",
            "session_weighting",
            "score_weighting",
            "mrr@10",
        ]
        print(best[columns].to_string(index=False))


if __name__ == "__main__":
    main()
