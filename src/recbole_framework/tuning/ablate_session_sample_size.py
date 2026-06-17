from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger
from src.recbole_framework.tuning.tune_session_models_full import (
    load_completed_run_ids,
    run_and_store,
)


DATASETS = [
    "adressa_recbole_sample",
    "globo_recbole_sample",
    "yoochoose_recbole_sample",
]

SAMPLE_SIZES = [500, 1000, 2000, 5000]

MODEL_SPECS = {
    "VS-KNN": {
        "model_class": VSKNNRecBole,
        "sample_column": "vsknn_sample_size",
        "popularity_weight_column": "vsknn_popularity_weight",
        "config_columns": [
            "vsknn_k",
            "vsknn_sample_size",
            "vsknn_popularity_weight",
        ],
    },
    "VSTAN": {
        "model_class": VSTANRecBole,
        "sample_column": "vstan_sample_size",
        "popularity_weight_column": "vstan_popularity_weight",
        "config_columns": [
            "vstan_k",
            "vstan_sample_size",
            "vstan_position_decay",
            "vstan_idf_weighting",
            "vstan_popularity_weight",
        ],
    },
}


FALLBACK_BASE_CONFIGS = {
    ("adressa_recbole_sample", "VS-KNN"): {
        "vsknn_k": 200,
        "vsknn_popularity_weight": 0.0,
    },
    ("adressa_recbole_sample", "VSTAN"): {
        "vstan_k": 100,
        "vstan_position_decay": 0.2,
        "vstan_idf_weighting": False,
        "vstan_popularity_weight": 0.0,
    },
    ("globo_recbole_sample", "VS-KNN"): {
        "vsknn_k": 500,
        "vsknn_popularity_weight": 0.0,
    },
    ("globo_recbole_sample", "VSTAN"): {
        "vstan_k": 200,
        "vstan_position_decay": 0.2,
        "vstan_idf_weighting": False,
        "vstan_popularity_weight": 0.0,
    },
    ("yoochoose_recbole_sample", "VS-KNN"): {
        "vsknn_k": 100,
        "vsknn_popularity_weight": 0.0,
    },
    ("yoochoose_recbole_sample", "VSTAN"): {
        "vstan_k": 500,
        "vstan_position_decay": 0.2,
        "vstan_idf_weighting": True,
        "vstan_popularity_weight": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an isolated sample-size ablation for session VS-KNN and VSTAN. "
            "Only sample_size is varied; all other hyperparameters are fixed to "
            "the current best unweighted configuration. This tests whether the "
            "newest-candidate sampling effect is caused by too small a candidate "
            "pool, especially on Globo."
        )
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=SAMPLE_SIZES,
        help="Sample sizes to test for both VS-KNN and VSTAN.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Session datasets to test.",
    )
    parser.add_argument(
        "--best-config-file",
        type=Path,
        default=None,
        help=(
            "Optional best_per_model.csv to derive fixed base configs from. "
            "Falls back to built-in configs from the latest 17.06 evaluation."
        ),
    )
    return parser.parse_args()


def parse_config_json(value: object) -> dict:
    if pd.isna(value) or value == "":
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def normalize_config(config: dict) -> dict:
    normalized = {}
    for key, value in config.items():
        if pd.isna(value):
            continue
        if isinstance(value, float) and value.is_integer():
            normalized[key] = int(value)
        else:
            normalized[key] = value
    return normalized


def load_best_base_configs(best_config_file: Path | None) -> dict[tuple[str, str], dict]:
    if best_config_file is None or not best_config_file.exists():
        return FALLBACK_BASE_CONFIGS.copy()

    best_configs: dict[tuple[str, str], dict] = {}
    best_df = pd.read_csv(best_config_file)

    for _, row in best_df.iterrows():
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", ""))
        if dataset not in DATASETS or model not in MODEL_SPECS:
            continue

        spec = MODEL_SPECS[model]
        config = parse_config_json(row.get("config_json"))

        for column in spec["config_columns"]:
            if column in row and not pd.isna(row[column]):
                config[column] = row[column]

        config[spec["popularity_weight_column"]] = 0.0
        config.pop(spec["sample_column"], None)
        best_configs[(dataset, model)] = normalize_config(config)

    fallback = FALLBACK_BASE_CONFIGS.copy()
    fallback.update(best_configs)
    return fallback


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    best_config_file = args.best_config_file or (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "analysis_results"
        / "structured_report"
        / "best_per_model.csv"
    )

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "session_sample_size_ablation_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "session_sample_size_ablation_log.csv"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_completed_run_ids(output_file)
    base_configs = load_best_base_configs(best_config_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running isolated session sample-size ablation")
    print("Reason: test whether newest-candidate sampling needs a larger pool.")
    print("Only sample_size changes; popularity_weight is fixed to 0.0.")
    print(f"Using device: {device}")
    print(f"Best config source: {best_config_file}")
    print(f"Output file: {output_file}")
    print(f"Already completed ablation runs: {len(completed_run_ids)}")

    for dataset_name in args.datasets:
        print(f"\n===== DATASET: {dataset_name} =====")

        for model_name, spec in MODEL_SPECS.items():
            base_config = base_configs.get((dataset_name, model_name))
            if not base_config:
                print(f"Skipping {model_name} on {dataset_name}: no base config found.")
                continue

            for sample_size in args.sample_sizes:
                config_updates = {
                    **base_config,
                    spec["sample_column"]: sample_size,
                }
                config_updates["ablation_name"] = "session_sample_size_sensitivity"
                config_updates["ablation_reason"] = (
                    "isolate newest-candidate sample-size effect"
                )

                print(f"Running {model_name} on {dataset_name}: {config_updates}")
                run_and_store(
                    output_file=output_file,
                    logger=logger,
                    completed_run_ids=completed_run_ids,
                    model_class=spec["model_class"],
                    model_name=model_name,
                    dataset_name=dataset_name,
                    config_updates=config_updates,
                    device=device,
                )

    print(f"\nSaved sample-size ablation results to: {output_file}")
    print(f"Saved sample-size ablation log to: {log_file}")


if __name__ == "__main__":
    main()
