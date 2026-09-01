"""Run model selection on validation data before one final test evaluation.

This runner replaces the exploratory workflow that stored a test result for
every hyperparameter configuration. During the tuning phase, RecBole's
``Trainer.fit`` returns the best validation result and the test loader is never
evaluated. During the final phase, one configuration per model and dataset is
selected by validation MRR@10 and frozen. Every model is refitted with the
primary seed 42. For BPR and GRU4Rec, seed 43 can be added as an optional
robustness check; it is not required for protocol completion and does not change
the primary seed-42 result.

Historical result files are left unchanged. By default, new files are written
below ``recbole_results/validation_first`` and every row records the evaluated
split. ``--output-dir`` can isolate a worker on another computer so that its
rows can be merged after both writers have stopped.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed

from src.recbole_framework.custom_models.session.popularity_recbole import (
    SessionDecayPopRecBole,
    SessionMostPopRecBole,
    SessionRecentPopRecBole,
)
from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNN
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole
from src.recbole_framework.custom_models.topn.decaypop_recbole import DecayPopRecBole
from src.recbole_framework.custom_models.topn.mostpop_recbole import MostPopRecBole
from src.recbole_framework.custom_models.topn.recentpop_recbole import RecentPopRecBole
from tools.measure_selected_session_beyond_accuracy import beyond_accuracy_metrics


OUTPUT_DIR = PROJECT_ROOT / "recbole_results" / "validation_first"
VALIDATION_FILE = OUTPUT_DIR / "validation_trials.csv"
FINAL_TEST_FILE = OUTPUT_DIR / "final_test_results.csv"
FINAL_TEST_SUMMARY_FILE = OUTPUT_DIR / "final_test_summary.csv"
PROTOCOL_VERSION = "validation_first_v6"
RANDOM_SEARCH_BUDGET = 12
MAX_BPR_EMBEDDING_SIZE = 256
PRIMARY_FINAL_EVALUATION_SEEDS = (42,)
OPTIONAL_ROBUSTNESS_SEEDS = (43,)
STOCHASTIC_MODELS = {"BPR", "GRU4Rec"}
FINAL_AGGREGATE_METRICS = (
    "hit@5",
    "hit@10",
    "ndcg@5",
    "ndcg@10",
    "mrr@5",
    "mrr@10",
    "coverage@10",
    "avg_recommendation_popularity@10",
    "recommendation_frequency_gini@10",
)

TOPN_DATASETS = ["movielens_recbole", "amazon_recbole"]
SESSION_DATASETS = [
    "adressa_recbole_sample",
    "globo_recbole_sample",
    "yoochoose_recbole_sample",
]

TOPN_MODELS = {
    "MostPop": MostPopRecBole,
    "RecentPop": RecentPopRecBole,
    "DecayPop": DecayPopRecBole,
    "BPR": BPR,
}
SESSION_MODELS = {
    "MostPop": SessionMostPopRecBole,
    "RecentPop": SessionRecentPopRecBole,
    "DecayPop": SessionDecayPopRecBole,
    "GRU4Rec": GRU4Rec,
    "VS-KNN": VSKNN,
    "VSTAN": VSTANRecBole,
}


def configure_output_dir(output_dir: Path | None) -> None:
    """Redirect all result files while preserving the historical default."""
    if output_dir is None:
        return

    resolved = output_dir
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()

    global OUTPUT_DIR, VALIDATION_FILE, FINAL_TEST_FILE, FINAL_TEST_SUMMARY_FILE
    OUTPUT_DIR = resolved
    VALIDATION_FILE = OUTPUT_DIR / "validation_trials.csv"
    FINAL_TEST_FILE = OUTPUT_DIR / "final_test_results.csv"
    FINAL_TEST_SUMMARY_FILE = OUTPUT_DIR / "final_test_summary.csv"


def fixed_budget_sample(
    candidates: list[dict], reference: dict, budget: int, seed: int
) -> list[dict]:
    """Return one reference plus a reproducible random sample of configurations."""
    unique = {serialise_config(config): config for config in candidates}
    reference_key = serialise_config(reference)
    unique.pop(reference_key, None)
    pool = [unique[key] for key in sorted(unique)]
    if budget < 1 or budget - 1 > len(pool):
        raise ValueError("Random-search budget is incompatible with the search space")
    sampled = random.Random(seed).sample(pool, budget - 1)
    return [reference, *sampled]


def topn_grid(model_name: str) -> list[dict]:
    if model_name == "MostPop":
        return [{}]
    if model_name == "RecentPop":
        return [
            {"window_days": value}
            for value in [1, 3, 7, 14, 30, 60, 90, 180]
        ]
    if model_name == "DecayPop":
        return [
            {"decay_lambda": value}
            for value in [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
        ]
    if model_name == "BPR":
        return [
            {
                "embedding_size": embedding_size,
                "learning_rate": learning_rate,
                "epochs": 50,
                "train_batch_size": 2048,
                "eval_batch_size": 2048,
                "train_neg_sample_args": {
                    "distribution": "uniform",
                    "sample_num": 1,
                    "alpha": 1.0,
                    "dynamic": False,
                    "candidate_num": 0,
                },
            }
            for embedding_size, learning_rate in product(
                [32, 64, 128, MAX_BPR_EMBEDDING_SIZE],
                [0.001, 0.0005, 0.0001],
            )
        ]
    raise KeyError(model_name)


def session_grid(model_name: str) -> list[dict]:
    if model_name == "MostPop":
        return [{}]
    if model_name == "RecentPop":
        return [
            {"recent_fraction": value}
            for value in [0.05, 0.10, 0.25, 0.50, 0.75]
        ]
    if model_name == "DecayPop":
        return [
            {"decay_half_life_days": value}
            for value in [0.25, 1.0, 7.0, 30.0, 90.0]
        ]
    if model_name == "VS-KNN":
        reference = {
            "neighbor_size": 100,
            "sample_size": 1000,
            "sampling": "recent",
            "similarity": "vec",
            "session_weighting": "div",
            "score_weighting": "div",
        }
        candidates = [
            {
                "neighbor_size": neighbours,
                "sample_size": candidate_limit,
                "sampling": "recent",
                "similarity": similarity,
                "session_weighting": session_weighting,
                "score_weighting": score_weighting,
            }
            for neighbours, candidate_limit, similarity, session_weighting, score_weighting
            in product(
                [50, 100, 200, 500, 1000, 1500],
                [500, 1000, 2500, 5000],
                ["vec", "cosine"],
                ["same", "linear", "div", "log", "quadratic"],
                ["same", "linear", "div", "log", "quadratic"],
            )
            if neighbours <= candidate_limit
        ]
        return fixed_budget_sample(
            candidates, reference, RANDOM_SEARCH_BUDGET, seed=4201
        )
    if model_name == "VSTAN":
        reference = {
            "vstan_k": 200,
            "vstan_sample_size": 2500,
            "vstan_position_decay": 0.1,
            "vstan_idf_weighting": True,
            "vstan_popularity_weight": 0.0,
        }
        candidates = [
            {
                "vstan_k": neighbours,
                "vstan_sample_size": candidate_limit,
                "vstan_position_decay": position_decay,
                "vstan_idf_weighting": idf_weighting,
                "vstan_popularity_weight": popularity_weight,
            }
            for neighbours, candidate_limit, position_decay, idf_weighting, popularity_weight
            in product(
                [100, 200, 500, 1000, 1500, 2000],
                [1000, 2500],
                [0.01, 0.05, 0.1, 0.2, 0.5],
                [True, False],
                [0.0, 0.5, 1.0],
            )
            if neighbours <= candidate_limit
        ]
        return fixed_budget_sample(
            candidates, reference, RANDOM_SEARCH_BUDGET, seed=4202
        )
    if model_name == "GRU4Rec":
        reference = {
            "hidden_size": 128,
            "learning_rate": 0.0005,
            "dropout_prob": 0.1,
            "epochs": 20,
            "num_layers": 1,
            "loss_type": "CE",
            "train_neg_sample_args": None,
            "train_batch_size": 2048,
            "eval_batch_size": 1024,
        }
        candidates = [
            {
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "dropout_prob": dropout,
                "epochs": epochs,
                "num_layers": 1,
                "loss_type": "CE",
                "train_neg_sample_args": None,
                "train_batch_size": train_batch_size,
                "eval_batch_size": 1024,
            }
            for hidden_size, learning_rate, dropout, epochs, train_batch_size in product(
                [64, 128, 256, 512],
                [0.0001, 0.0005, 0.001, 0.003],
                [0.0, 0.1, 0.2, 0.4],
                [10, 20, 30],
                [512, 1024, 2048],
            )
        ]
        return fixed_budget_sample(
            candidates, reference, RANDOM_SEARCH_BUDGET, seed=4203
        )
    raise KeyError(model_name)


def serialise_config(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def run_id(scenario: str, dataset: str, model: str, config: dict) -> str:
    return "::".join(
        [PROTOCOL_VERSION, scenario, dataset, model, serialise_config(config)]
    )


def build_config(
    scenario: str,
    model_class,
    dataset_name: str,
    updates: dict,
    device: str,
    checkpoint_dir: Path | None = None,
    seed: int = 42,
) -> Config:
    config_dict = {
        "model": model_class,
        "dataset": dataset_name,
        "data_path": str(PROJECT_ROOT / "data" / "recbole"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "epochs": 1,
        "train_batch_size": 2048,
        "eval_batch_size": 2048 if scenario == "topn" else 1024,
        "topk": [5, 10],
        "metrics": ["Hit", "NDCG", "MRR"],
        "valid_metric": "MRR@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO",
            "mode": "full",
        },
        "seed": seed,
        "reproducibility": True,
        "device": device,
        "show_progress": False,
    }
    if scenario == "session":
        config_dict["MAX_ITEM_LIST_LENGTH"] = 20
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_dict["checkpoint_dir"] = str(checkpoint_dir)
    config_dict.update(updates)
    return Config(model=model_class, config_dict=config_dict)


def prefixed_metrics(prefix: str, metrics: dict | None) -> dict:
    return {
        f"{prefix}_{key}": value
        for key, value in dict(metrics or {}).items()
    }


def load_locally_created_checkpoint(trainer: Trainer, model, device: str) -> None:
    """Load the trusted checkpoint created moments earlier by this process.

    RecBole 1.2.0 calls ``torch.load`` without the ``weights_only`` argument.
    PyTorch 2.6 changed that argument's default to ``True``, which cannot read
    RecBole's complete checkpoint dictionary. Explicitly allowing this locally
    created file preserves RecBole's previous checkpoint-loading behaviour.
    """
    checkpoint = torch.load(
        trainer.saved_model_file,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))


def upsert_result(path: Path, result: dict, key_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.read_csv(path, low_memory=False).to_dict("records")
        rows = [row for row in rows if str(row.get(key_column)) != str(result[key_column])]
    else:
        rows = []
    rows.append(result)
    pd.DataFrame(rows).to_csv(path, index=False)


def successful_ids(path: Path, key_column: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "status" not in frame or key_column not in frame:
        return set()
    return set(
        frame.loc[frame["status"].eq("success"), key_column]
        .dropna()
        .astype(str)
    )


def run_validation_trial(
    scenario: str,
    dataset_name: str,
    model_name: str,
    updates: dict,
    device: str,
) -> dict:
    model_class = (TOPN_MODELS if scenario == "topn" else SESSION_MODELS)[model_name]
    identifier = run_id(scenario, dataset_name, model_name, updates)
    started = time.perf_counter()
    try:
        config = build_config(
            scenario, model_class, dataset_name, updates, device
        )
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, dataset)
        init_seed(config["seed"], config["reproducibility"])
        model = model_class(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            verbose=False,
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "evaluated_split": "validation",
            "run_id": identifier,
            "scenario": scenario,
            "dataset": dataset_name,
            "model": model_name,
            "device": device,
            "best_valid_score": best_valid_score,
            **prefixed_metrics("valid", best_valid_result),
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "success",
            "error": "",
        }
    except Exception as error:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "evaluated_split": "validation",
            "run_id": identifier,
            "scenario": scenario,
            "dataset": dataset_name,
            "model": model_name,
            "device": device,
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def expected_trials(
    scenarios: list[str], datasets: set[str] | None, models: set[str] | None
):
    for scenario in scenarios:
        dataset_names = TOPN_DATASETS if scenario == "topn" else SESSION_DATASETS
        model_classes = TOPN_MODELS if scenario == "topn" else SESSION_MODELS
        grid_function = topn_grid if scenario == "topn" else session_grid
        for dataset_name in dataset_names:
            if datasets and dataset_name not in datasets:
                continue
            for model_name in model_classes:
                if models and model_name not in models:
                    continue
                for updates in grid_function(model_name):
                    yield scenario, dataset_name, model_name, updates


def run_tuning(
    scenarios: list[str],
    datasets: set[str] | None,
    models: set[str] | None,
    device: str,
) -> None:
    completed = successful_ids(VALIDATION_FILE, "run_id")
    for scenario, dataset_name, model_name, updates in expected_trials(
        scenarios, datasets, models
    ):
        identifier = run_id(scenario, dataset_name, model_name, updates)
        if identifier in completed:
            print(f"Skipping completed validation run: {identifier}", flush=True)
            continue
        print(
            f"Validation run: {scenario}, {dataset_name}, {model_name}, {updates}",
            flush=True,
        )
        result = run_validation_trial(
            scenario, dataset_name, model_name, updates, device
        )
        upsert_result(VALIDATION_FILE, result, "run_id")
        print(
            f"Finished: status={result['status']}, "
            f"valid_mrr@10={result.get('valid_mrr@10')}",
            flush=True,
        )


def select_validation_winners(
    scenarios: list[str], datasets: set[str] | None, models: set[str] | None
) -> pd.DataFrame:
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"No validation results found: {VALIDATION_FILE}")
    frame = pd.read_csv(VALIDATION_FILE, low_memory=False)
    frame = frame[
        frame["protocol_version"].eq(PROTOCOL_VERSION)
        & frame["evaluated_split"].eq("validation")
        & frame["status"].eq("success")
        & frame["scenario"].isin(scenarios)
    ].copy()
    if datasets:
        frame = frame[frame["dataset"].isin(datasets)]
    if models:
        frame = frame[frame["model"].isin(models)]
    eligible_ids = {
        run_id(scenario, dataset_name, model_name, updates)
        for scenario, dataset_name, model_name, updates in expected_trials(
            scenarios, datasets, models
        )
    }
    frame = frame[frame["run_id"].isin(eligible_ids)]

    frame["valid_mrr@10"] = pd.to_numeric(frame["valid_mrr@10"], errors="coerce")
    frame["runtime_seconds"] = pd.to_numeric(
        frame["runtime_seconds"], errors="coerce"
    ).fillna(float("inf"))
    frame = frame.dropna(subset=["valid_mrr@10"])
    ranked = frame.sort_values(
        ["scenario", "dataset", "model", "valid_mrr@10", "runtime_seconds", "run_id"],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    )
    return ranked.groupby(
        ["scenario", "dataset", "model"], as_index=False, sort=True
    ).head(1)


def ensure_tuning_complete(
    scenarios: list[str], datasets: set[str] | None, models: set[str] | None
) -> None:
    completed = successful_ids(VALIDATION_FILE, "run_id")
    expected = {
        run_id(scenario, dataset_name, model_name, updates)
        for scenario, dataset_name, model_name, updates in expected_trials(
            scenarios, datasets, models
        )
    }
    missing = sorted(expected - completed)
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(
            f"Final testing is blocked: {len(missing)} validation runs are missing.\n"
            f"First missing runs:\n{preview}"
        )


def final_seeds(
    model_name: str, include_optional_robustness: bool = False
) -> tuple[int, ...]:
    """Return required primary seeds and, when requested, one robustness seed."""
    seeds = PRIMARY_FINAL_EVALUATION_SEEDS
    if include_optional_robustness and model_name in STOCHASTIC_MODELS:
        seeds += OPTIONAL_ROBUSTNESS_SEEDS
    return seeds


def optional_final_seeds(model_name: str) -> tuple[int, ...]:
    """Return non-required seeds that may be run as a sensitivity check."""
    return OPTIONAL_ROBUSTNESS_SEEDS if model_name in STOCHASTIC_MODELS else ()


def final_test_id(scenario: str, dataset: str, model: str, seed: int) -> str:
    return "::".join(
        [PROTOCOL_VERSION, "test", scenario, dataset, model, f"seed={seed}"]
    )


def run_final_test(row: pd.Series, device: str, seed: int) -> dict:
    scenario = str(row["scenario"])
    dataset_name = str(row["dataset"])
    model_name = str(row["model"])
    updates = json.loads(str(row["config_json"]))
    model_class = (TOPN_MODELS if scenario == "topn" else SESSION_MODELS)[model_name]
    identifier = final_test_id(scenario, dataset_name, model_name, seed)
    checkpoint_dir = (
        OUTPUT_DIR
        / "checkpoints"
        / scenario
        / dataset_name
        / model_name
        / f"seed_{seed}"
    )
    started = time.perf_counter()
    try:
        config = build_config(
            scenario,
            model_class,
            dataset_name,
            updates,
            device,
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
            "evaluated_split": "test",
            "final_test_id": identifier,
            "selected_validation_run_id": row["run_id"],
            "scenario": scenario,
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "device": device,
            "selection_valid_mrr@10": row["valid_mrr@10"],
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
            "evaluated_split": "test",
            "final_test_id": identifier,
            "selected_validation_run_id": row["run_id"],
            "scenario": scenario,
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "device": device,
            "selection_valid_mrr@10": row["valid_mrr@10"],
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def build_final_test_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["scenario", "dataset", "model"]
    for (scenario, dataset, model), group in frame.groupby(
        group_columns, sort=True
    ):
        expected_seeds = final_seeds(str(model))
        optional_seeds = optional_final_seeds(str(model))
        completed_seeds = sorted(set(pd.to_numeric(group["seed"]).astype(int)))
        numeric_seeds = pd.to_numeric(group["seed"], errors="coerce")
        primary_group = group[numeric_seeds.isin(expected_seeds)]
        optional_group = group[numeric_seeds.isin(optional_seeds)]
        completed_primary = sorted(set(completed_seeds).intersection(expected_seeds))
        completed_optional = sorted(set(completed_seeds).intersection(optional_seeds))
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "dataset": dataset,
            "model": model,
            "device": ",".join(sorted(set(group["device"].astype(str)))),
            "seed_count": len(completed_seeds),
            "seeds": ",".join(str(seed) for seed in completed_seeds),
            "expected_seed_count": len(expected_seeds),
            "primary_seed_count": len(completed_primary),
            "primary_seeds": ",".join(str(seed) for seed in completed_primary),
            "optional_seed_count": len(completed_optional),
            "optional_seeds": ",".join(str(seed) for seed in completed_optional),
            "status": (
                "complete"
                if completed_primary == list(expected_seeds)
                else "incomplete"
            ),
            "config_json": str(group.iloc[0]["config_json"]),
        }
        for metric in FINAL_AGGREGATE_METRICS:
            values = (
                pd.to_numeric(primary_group[metric], errors="coerce").dropna()
                if metric in primary_group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            optional_values = (
                pd.to_numeric(optional_group[metric], errors="coerce").dropna()
                if metric in optional_group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric}_optional_mean"] = (
                float(optional_values.mean()) if len(optional_values) else None
            )
            row[f"{metric}_optional_difference_from_primary"] = (
                float(optional_values.mean() - values.mean())
                if len(optional_values) and len(values)
                else None
            )
        runtimes = pd.to_numeric(
            primary_group["runtime_seconds"], errors="coerce"
        ).dropna()
        row["runtime_seconds_median"] = (
            float(runtimes.median()) if len(runtimes) else None
        )
        row["runtime_seconds_min"] = float(runtimes.min()) if len(runtimes) else None
        row["runtime_seconds_max"] = float(runtimes.max()) if len(runtimes) else None
        optional_runtimes = pd.to_numeric(
            optional_group["runtime_seconds"], errors="coerce"
        ).dropna()
        row["runtime_seconds_optional_mean"] = (
            float(optional_runtimes.mean()) if len(optional_runtimes) else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_final_test_summary() -> None:
    if not FINAL_TEST_FILE.exists():
        return
    frame = pd.read_csv(FINAL_TEST_FILE, low_memory=False)
    frame = frame[
        frame["protocol_version"].eq(PROTOCOL_VERSION)
        & frame["evaluated_split"].eq("test")
        & frame["status"].eq("success")
    ].copy()
    if frame.empty:
        return
    summary = build_final_test_summary(frame)
    FINAL_TEST_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FINAL_TEST_SUMMARY_FILE, index=False)


def run_final_tests(
    scenarios: list[str],
    datasets: set[str] | None,
    models: set[str] | None,
    include_optional_robustness: bool = False,
) -> None:
    ensure_tuning_complete(scenarios, datasets, models)
    winners = select_validation_winners(scenarios, datasets, models)
    completed = successful_ids(FINAL_TEST_FILE, "final_test_id")
    for _, row in winners.iterrows():
        model_name = str(row["model"])
        for seed in final_seeds(model_name, include_optional_robustness):
            identifier = final_test_id(
                str(row["scenario"]), str(row["dataset"]), model_name, seed
            )
            if identifier in completed:
                print(f"Skipping completed final test: {identifier}", flush=True)
                continue
            print(
                f"Final test: {row['scenario']}, {row['dataset']}, "
                f"{model_name}, seed={seed}",
                flush=True,
            )
            result = run_final_test(row, "cpu", seed)
            upsert_result(FINAL_TEST_FILE, result, "final_test_id")
            print(
                f"Finished final test: status={result['status']}, "
                f"test_mrr@10={result.get('mrr@10')}",
                flush=True,
            )
    write_final_test_summary()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["tune", "test", "all"], default="tune"
    )
    parser.add_argument(
        "--scenario", choices=["topn", "session", "both"], default="both"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(TOPN_DATASETS + SESSION_DATASETS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(set(TOPN_MODELS) | set(SESSION_MODELS)),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="device used for validation tuning; final test runs always use CPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "separate result directory, relative to the project root unless "
            "an absolute path is supplied"
        ),
    )
    parser.add_argument(
        "--include-optional-robustness-seed",
        action="store_true",
        help=(
            "also run seed 43 for BPR and GRU4Rec; seed 42 remains the primary "
            "result and seed 43 is not required for protocol completion"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_output_dir(args.output_dir)
    if (
        args.phase in {"tune", "all"}
        and args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested for tuning but is not available.")
    scenarios = (
        ["topn", "session"] if args.scenario == "both" else [args.scenario]
    )
    datasets = set(args.datasets) if args.datasets else None
    models = set(args.models) if args.models else None

    if args.phase in {"tune", "all"}:
        run_tuning(scenarios, datasets, models, args.device)
    if args.phase in {"test", "all"}:
        run_final_tests(
            scenarios,
            datasets,
            models,
            include_optional_robustness=args.include_optional_robustness_seed,
        )


if __name__ == "__main__":
    main()
