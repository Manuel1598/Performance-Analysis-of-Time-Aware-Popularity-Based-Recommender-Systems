"""Measure beyond-accuracy metrics for the selected session configurations.

This script reruns exactly one selected configuration per model and dataset.  It
does not perform hyperparameter tuning and writes to a dedicated result file so
that the historical tuning logs remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = (
    PROJECT_ROOT
    / "recbole_results"
    / "selected_session_beyond_accuracy"
    / "selected_session_metrics.csv"
)

MODEL_CLASSES = {
    "MostPop": SessionMostPopRecBole,
    "RecentPop": SessionRecentPopRecBole,
    "DecayPop": SessionDecayPopRecBole,
    "VS-KNN": VSKNN,
    "VSTAN": VSTANRecBole,
    "GRU4Rec": GRU4Rec,
}

SELECTED_CONFIGS = {
    "adressa_recbole_sample": {
        "MostPop": {},
        "RecentPop": {"recent_fraction": 0.25},
        "DecayPop": {"decay_half_life_days": 0.25},
        "VS-KNN": {
            "neighbor_size": 200,
            "sample_size": 500,
            "sampling": "recent",
            "similarity": "vec",
            "session_weighting": "div",
            "score_weighting": "div",
        },
        "VSTAN": {
            "vstan_k": 100,
            "vstan_sample_size": 500,
            "vstan_position_decay": 0.2,
            "vstan_idf_weighting": False,
            "vstan_popularity_weight": 0.0,
        },
        "GRU4Rec": {
            "hidden_size": 256,
            "learning_rate": 0.001,
            "dropout_prob": 0.1,
            "epochs": 20,
            "num_layers": 1,
            "loss_type": "CE",
            "train_neg_sample_args": None,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
        },
    },
    "globo_recbole_sample": {
        "MostPop": {},
        "RecentPop": {"recent_fraction": 0.5},
        "DecayPop": {"decay_half_life_days": 30.0},
        "VS-KNN": {
            "neighbor_size": 100,
            "sample_size": 1000,
            "sampling": "recent",
            "similarity": "vec",
            "session_weighting": "div",
            "score_weighting": "div",
        },
        "VSTAN": {
            "vstan_k": 200,
            "vstan_sample_size": 500,
            "vstan_position_decay": 0.2,
            "vstan_idf_weighting": False,
            "vstan_popularity_weight": 0.0,
        },
        "GRU4Rec": {
            "hidden_size": 256,
            "learning_rate": 0.001,
            "dropout_prob": 0.1,
            "epochs": 20,
            "num_layers": 1,
            "loss_type": "CE",
            "train_neg_sample_args": None,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
        },
    },
    "yoochoose_recbole_sample": {
        "MostPop": {},
        "RecentPop": {"recent_fraction": 0.5},
        "DecayPop": {"decay_half_life_days": 30.0},
        "VS-KNN": {
            "neighbor_size": 100,
            "sample_size": 500,
            "sampling": "recent",
            "similarity": "vec",
            "session_weighting": "div",
            "score_weighting": "quadratic",
        },
        "VSTAN": {
            "vstan_k": 200,
            "vstan_sample_size": 500,
            "vstan_position_decay": 0.2,
            "vstan_idf_weighting": True,
            "vstan_popularity_weight": 0.0,
        },
        "GRU4Rec": {
            "hidden_size": 256,
            "learning_rate": 0.001,
            "dropout_prob": 0.1,
            "epochs": 20,
            "num_layers": 1,
            "loss_type": "CE",
            "train_neg_sample_args": None,
            "train_batch_size": 2048,
            "eval_batch_size": 2048,
        },
    },
}


def build_config(model_class, dataset_name: str, updates: dict, device: str) -> Config:
    config_dict = {
        "model": model_class,
        "dataset": dataset_name,
        "data_path": str(PROJECT_ROOT / "data" / "recbole"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "MAX_ITEM_LIST_LENGTH": 20,
        "epochs": 1,
        "train_batch_size": 2048,
        "eval_batch_size": 1024,
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
    config_dict.update(updates)
    return Config(model=model_class, config_dict=config_dict)


def interaction_and_history(batch_data):
    if isinstance(batch_data, tuple):
        interaction = batch_data[0]
        history = batch_data[1] if len(batch_data) > 1 else None
        return interaction, history
    return batch_data, None


def gini_coefficient(counts: torch.Tensor) -> float:
    """Measure concentration over all catalogue items, including zero counts."""
    values = counts.detach().to(dtype=torch.float64, device="cpu").flatten()
    total = values.sum()
    if values.numel() == 0 or total <= 0:
        return 0.0
    values, _ = torch.sort(values)
    n_values = values.numel()
    ranks = torch.arange(1, n_values + 1, dtype=torch.float64)
    coefficient = ((2 * ranks - n_values - 1) * values).sum()
    return float(coefficient / (n_values * total))


def beyond_accuracy_metrics(model, test_data, train_data, config, top_k: int = 10):
    model.eval()
    item_field = config["ITEM_ID_FIELD"]
    n_items_with_padding = train_data.dataset.num(item_field)
    catalogue_size = max(n_items_with_padding - 1, 0)
    train_ids = train_data.dataset.inter_feat[item_field].long().cpu()
    popularity = torch.bincount(train_ids, minlength=n_items_with_padding).float()

    recommended_items: set[int] = set()
    recommendation_frequency = torch.zeros(n_items_with_padding, dtype=torch.long)
    popularity_sum = 0.0
    recommendation_count = 0

    started = time.perf_counter()
    with torch.no_grad():
        for batch_data in test_data:
            interaction, history_index = interaction_and_history(batch_data)
            interaction = interaction.to(config["device"])
            scores = model.full_sort_predict(interaction)
            scores = scores.view(-1, n_items_with_padding)

            # RecBole reserves item 0 for padding; it is not a recommendable item.
            scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")

            top_items = torch.topk(scores, k=top_k, dim=1).indices.cpu()
            flat_items = top_items.reshape(-1)
            recommended_items.update(int(item) for item in flat_items.tolist())
            recommendation_frequency.index_add_(
                0, flat_items, torch.ones_like(flat_items, dtype=torch.long)
            )
            popularity_sum += float(popularity[flat_items].sum().item())
            recommendation_count += int(flat_items.numel())

    return {
        f"coverage@{top_k}": (
            len(recommended_items) / catalogue_size if catalogue_size else 0.0
        ),
        f"avg_recommendation_popularity@{top_k}": (
            popularity_sum / recommendation_count if recommendation_count else 0.0
        ),
        f"recommendation_frequency_gini@{top_k}": gini_coefficient(
            recommendation_frequency[1:]
        ),
        "unique_recommended_items@10": len(recommended_items),
        "catalogue_size": catalogue_size,
        "recommendation_count": recommendation_count,
        "extra_metrics_runtime_seconds": round(time.perf_counter() - started, 2),
    }


def run_selected(dataset_name: str, model_name: str, device: str) -> dict:
    model_class = MODEL_CLASSES[model_name]
    updates = SELECTED_CONFIGS[dataset_name][model_name]
    config = build_config(model_class, dataset_name, updates, device)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    started = time.perf_counter()
    trainer.fit(train_data, valid_data, saved=False, verbose=False)
    ranking_metrics = dict(trainer.evaluate(test_data, load_best_model=False))
    beyond_metrics = beyond_accuracy_metrics(
        model, test_data, train_data, config, top_k=10
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "implementation": "audited" if model_name == "VS-KNN" else "current",
        "device": device,
        **ranking_metrics,
        **beyond_metrics,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "config_json": json.dumps(updates, sort_keys=True),
        "status": "success",
        "error": "",
    }


def result_key(row: dict) -> tuple[str, str]:
    return str(row["dataset"]), str(row["model"])


def save_result(result: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        rows = pd.read_csv(OUTPUT_FILE).to_dict("records")
        rows = [row for row in rows if result_key(row) != result_key(result)]
    else:
        rows = []
    rows.append(result)
    pd.DataFrame(rows).sort_values(["dataset", "model"]).to_csv(
        OUTPUT_FILE, index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(SELECTED_CONFIGS),
        default=sorted(SELECTED_CONFIGS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_CLASSES),
        default=sorted(MODEL_CLASSES)
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    completed = set()
    if OUTPUT_FILE.exists() and not args.force:
        prior = pd.read_csv(OUTPUT_FILE)
        completed = set(
            zip(prior.loc[prior["status"] == "success", "dataset"],
                prior.loc[prior["status"] == "success", "model"])
        )

    print(f"device={device}")
    print(f"output={OUTPUT_FILE}")
    for dataset_name in args.datasets:
        for model_name in args.models:
            if (dataset_name, model_name) in completed:
                print(f"skip {dataset_name} / {model_name}")
                continue
            print(f"run {dataset_name} / {model_name}", flush=True)
            try:
                result = run_selected(dataset_name, model_name, device)
            except Exception as exc:
                result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "implementation": (
                        "audited" if model_name == "VS-KNN" else "current"
                    ),
                    "device": device,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "config_json": json.dumps(
                        SELECTED_CONFIGS[dataset_name][model_name], sort_keys=True
                    ),
                }
            save_result(result)
            print(
                f"status={result['status']} mrr@10={result.get('mrr@10')} "
                f"coverage@10={result.get('coverage@10')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
