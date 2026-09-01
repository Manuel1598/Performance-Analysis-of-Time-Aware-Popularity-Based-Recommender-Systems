"""Test sensitivity to the maximum retained session-prefix length.

The script starts from the configuration selected by validation MRR@10 in the
main validation-first protocol. It refits that frozen configuration with maximum
prefix lengths 10, 20, and 50 and evaluates the test split once per final seed.
Every required ablation uses seed 42. GRU4Rec is not repeated automatically;
this keeps the ablation within the same primary one-seed budget. All runs use CPU
so their recorded runtimes
follow the same hardware protocol as the main final comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

from tools.measure_selected_session_beyond_accuracy import beyond_accuracy_metrics
from tools.run_validation_first_experiments import (
    FINAL_AGGREGATE_METRICS,
    OUTPUT_DIR,
    PROTOCOL_VERSION as SELECTION_PROTOCOL_VERSION,
    SESSION_DATASETS,
    SESSION_MODELS,
    build_config,
    ensure_tuning_complete,
    final_seeds,
    load_locally_created_checkpoint,
    prefixed_metrics,
    select_validation_winners,
    serialise_config,
    successful_ids,
    upsert_result,
)


DEFAULT_MODELS = ("GRU4Rec", "VS-KNN", "VSTAN")
DEFAULT_LENGTHS = (10, 20, 50)
PROTOCOL_VERSION = f"session_length_v2_from_{SELECTION_PROTOCOL_VERSION}"
OUTPUT_DIR_ABLATION = OUTPUT_DIR / "session_sequence_length_ablation"
RAW_OUTPUT_FILE = OUTPUT_DIR_ABLATION / "raw_test_results.csv"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR_ABLATION / "summary.csv"


def ablation_id(dataset: str, model: str, max_length: int, seed: int) -> str:
    return "::".join(
        [
            PROTOCOL_VERSION,
            dataset,
            model,
            f"max_length={max_length}",
            f"seed={seed}",
        ]
    )


def run_configuration(
    winner: pd.Series,
    max_length: int,
    seed: int,
) -> dict:
    dataset_name = str(winner["dataset"])
    model_name = str(winner["model"])
    model_class = SESSION_MODELS[model_name]
    selected_updates = json.loads(str(winner["config_json"]))
    updates = {**selected_updates, "MAX_ITEM_LIST_LENGTH": max_length}
    identifier = ablation_id(dataset_name, model_name, max_length, seed)
    checkpoint_dir = (
        OUTPUT_DIR_ABLATION
        / "checkpoints"
        / dataset_name
        / model_name
        / f"length_{max_length}"
        / f"seed_{seed}"
    )
    started = time.perf_counter()

    try:
        config = build_config(
            "session",
            model_class,
            dataset_name,
            updates,
            "cpu",
            checkpoint_dir=checkpoint_dir,
            seed=seed,
        )
        data_preparation_started = time.perf_counter()
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        data_preparation_runtime = time.perf_counter() - data_preparation_started

        init_seed(config["seed"], config["reproducibility"])
        model = model_class(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)
        training_started = time.perf_counter()
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            verbose=False,
        )
        training_runtime = time.perf_counter() - training_started

        load_locally_created_checkpoint(trainer, model, config["device"])
        evaluation_started = time.perf_counter()
        test_result = dict(trainer.evaluate(test_data, load_best_model=False))
        evaluation_runtime = time.perf_counter() - evaluation_started
        additional_metrics = beyond_accuracy_metrics(
            model, test_data, train_data, config, top_k=10
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
            "evaluated_split": "test",
            "ablation_id": identifier,
            "selected_validation_run_id": winner["run_id"],
            "dataset": dataset_name,
            "model": model_name,
            "max_item_list_length": max_length,
            "seed": seed,
            "device": "cpu",
            "selection_valid_mrr@10": winner["valid_mrr@10"],
            "refit_best_valid_score": best_valid_score,
            **prefixed_metrics("refit_valid", best_valid_result),
            **test_result,
            **additional_metrics,
            "data_preparation_runtime_seconds": round(data_preparation_runtime, 2),
            "training_runtime_seconds": round(training_runtime, 2),
            "evaluation_runtime_seconds": round(evaluation_runtime, 2),
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "success",
            "error": "",
        }
    except Exception as error:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
            "evaluated_split": "test",
            "ablation_id": identifier,
            "selected_validation_run_id": winner["run_id"],
            "dataset": dataset_name,
            "model": model_name,
            "max_item_list_length": max_length,
            "seed": seed,
            "device": "cpu",
            "selection_valid_mrr@10": winner["valid_mrr@10"],
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def write_summary() -> None:
    if not RAW_OUTPUT_FILE.exists():
        return
    frame = pd.read_csv(RAW_OUTPUT_FILE, low_memory=False)
    frame = frame[
        frame["protocol_version"].eq(PROTOCOL_VERSION)
        & frame["status"].eq("success")
    ].copy()
    if frame.empty:
        return

    rows: list[dict] = []
    for (dataset, model, max_length), group in frame.groupby(
        ["dataset", "model", "max_item_list_length"], sort=True
    ):
        expected = list(final_seeds(str(model)))
        completed = sorted(set(pd.to_numeric(group["seed"]).astype(int)))
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": dataset,
            "model": model,
            "max_item_list_length": int(max_length),
            "device": "cpu",
            "seed_count": len(completed),
            "seeds": ",".join(str(seed) for seed in completed),
            "expected_seed_count": len(expected),
            "status": "complete" if completed == expected else "incomplete",
            "config_json": str(group.iloc[0]["config_json"]),
        }
        for metric in FINAL_AGGREGATE_METRICS:
            values = (
                pd.to_numeric(group[metric], errors="coerce").dropna()
                if metric in group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        runtimes = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
        row["runtime_seconds_median"] = (
            float(runtimes.median()) if len(runtimes) else None
        )
        row["runtime_seconds_min"] = float(runtimes.min()) if len(runtimes) else None
        row["runtime_seconds_max"] = float(runtimes.max()) if len(runtimes) else None
        rows.append(row)

    SUMMARY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_OUTPUT_FILE, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SESSION_DATASETS,
        default=SESSION_DATASETS,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(length < 1 for length in args.lengths):
        raise ValueError("Maximum sequence lengths must be positive integers")

    datasets = set(args.datasets)
    models = set(args.models)
    ensure_tuning_complete(["session"], datasets, models)
    winners = select_validation_winners(["session"], datasets, models)
    expected_winners = len(datasets) * len(models)
    if len(winners) != expected_winners:
        raise RuntimeError(
            f"Expected {expected_winners} validation winners, found {len(winners)}"
        )

    completed = successful_ids(RAW_OUTPUT_FILE, "ablation_id")
    for _, winner in winners.iterrows():
        model_name = str(winner["model"])
        for max_length in args.lengths:
            for seed in final_seeds(model_name):
                identifier = ablation_id(
                    str(winner["dataset"]), model_name, max_length, seed
                )
                if identifier in completed:
                    print(f"Skipping completed ablation run: {identifier}", flush=True)
                    continue
                print(f"Ablation run: {identifier}", flush=True)
                result = run_configuration(winner, max_length, seed)
                upsert_result(RAW_OUTPUT_FILE, result, "ablation_id")
                print(
                    f"Finished: status={result['status']}, "
                    f"test_mrr@10={result.get('mrr@10')}",
                    flush=True,
                )
    write_summary()


if __name__ == "__main__":
    main()
