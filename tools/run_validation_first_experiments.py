"""Run model selection on validation data before one final test evaluation.

This runner replaces the exploratory workflow that stored a test result for
every hyperparameter configuration. During the tuning phase, RecBole's
``Trainer.fit`` returns the best validation result and the test loader is never
evaluated. During the final phase, one configuration per model and dataset is
selected by validation MRR@10, retrained, and evaluated once on the test split.

Historical result files are left unchanged. New files are written below
``recbole_results/validation_first`` and every row records the evaluated split.
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
PROTOCOL_VERSION = "validation_first_v2"
RANDOM_SEARCH_BUDGET = 12

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
                "train_neg_sample_args": {
                    "distribution": "uniform",
                    "sample_num": 1,
                    "alpha": 1.0,
                    "dynamic": False,
                    "candidate_num": 0,
                },
            }
            for embedding_size, learning_rate in product(
                [32, 64, 128, 256, 512], [0.001, 0.0005, 0.0001]
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
            "eval_batch_size": 2048,
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
                "train_batch_size": 2048,
                "eval_batch_size": 2048,
            }
            for hidden_size, learning_rate, dropout, epochs in product(
                [64, 128, 256, 512],
                [0.0001, 0.0005, 0.001, 0.003],
                [0.0, 0.1, 0.2, 0.4],
                [10, 20, 30],
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
        "seed": 42,
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
        dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, dataset)
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


def final_test_id(scenario: str, dataset: str, model: str) -> str:
    return "::".join([PROTOCOL_VERSION, "test", scenario, dataset, model])


def run_final_test(row: pd.Series, device: str) -> dict:
    scenario = str(row["scenario"])
    dataset_name = str(row["dataset"])
    model_name = str(row["model"])
    updates = json.loads(str(row["config_json"]))
    model_class = (TOPN_MODELS if scenario == "topn" else SESSION_MODELS)[model_name]
    identifier = final_test_id(scenario, dataset_name, model_name)
    checkpoint_dir = OUTPUT_DIR / "checkpoints" / scenario / dataset_name / model_name
    started = time.perf_counter()
    try:
        config = build_config(
            scenario,
            model_class,
            dataset_name,
            updates,
            device,
            checkpoint_dir=checkpoint_dir,
        )
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = model_class(config, train_data.dataset).to(config["device"])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            verbose=False,
        )
        load_locally_created_checkpoint(trainer, model, config["device"])
        test_result = dict(trainer.evaluate(test_data, load_best_model=False))
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
            "device": device,
            "selection_valid_mrr@10": row["valid_mrr@10"],
            "refit_best_valid_score": best_valid_score,
            **prefixed_metrics("refit_valid", best_valid_result),
            **test_result,
            **additional_metrics,
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
            "device": device,
            "selection_valid_mrr@10": row["valid_mrr@10"],
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "config_json": serialise_config(updates),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def run_final_tests(
    scenarios: list[str],
    datasets: set[str] | None,
    models: set[str] | None,
    device: str,
) -> None:
    ensure_tuning_complete(scenarios, datasets, models)
    winners = select_validation_winners(scenarios, datasets, models)
    completed = successful_ids(FINAL_TEST_FILE, "final_test_id")
    for _, row in winners.iterrows():
        identifier = final_test_id(row["scenario"], row["dataset"], row["model"])
        if identifier in completed:
            print(f"Skipping completed final test: {identifier}", flush=True)
            continue
        print(
            f"Final test: {row['scenario']}, {row['dataset']}, {row['model']}",
            flush=True,
        )
        result = run_final_test(row, device)
        upsert_result(FINAL_TEST_FILE, result, "final_test_id")
        print(
            f"Finished final test: status={result['status']}, "
            f"test_mrr@10={result.get('mrr@10')}",
            flush=True,
        )


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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    scenarios = (
        ["topn", "session"] if args.scenario == "both" else [args.scenario]
    )
    datasets = set(args.datasets) if args.datasets else None
    models = set(args.models) if args.models else None

    if args.phase in {"tune", "all"}:
        run_tuning(scenarios, datasets, models, args.device)
    if args.phase in {"test", "all"}:
        run_final_tests(scenarios, datasets, models, args.device)


if __name__ == "__main__":
    main()
