from itertools import product
from pathlib import Path
import time

import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.topn.mostpop_recbole import MostPopRecBole
from src.recbole_framework.custom_models.topn.recentpop_recbole import RecentPopRecBole
from src.recbole_framework.custom_models.topn.decaypop_recbole import DecayPopRecBole
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger


ENABLE_EXTRA_METRICS = True


def make_run_id(model_name: str, dataset_name: str, config_updates: dict) -> str:
    config_string = ExperimentLogger.serialize_config(config_updates)
    return f"{dataset_name}::{model_name}::{config_string}"


def load_completed_run_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    df = pd.read_csv(output_file)

    if "run_id" not in df.columns or "status" not in df.columns:
        return set()

    successful_runs = df[df["status"] == "success"]
    return set(successful_runs["run_id"].dropna().astype(str))


def calculate_extra_metrics(model, test_data, train_data, config, top_k: int = 10) -> dict:
    model.eval()

    item_field = config["ITEM_ID_FIELD"]
    n_items = train_data.dataset.num(item_field)

    train_item_ids = train_data.dataset.inter_feat[item_field].long().cpu()
    item_popularity = torch.bincount(train_item_ids, minlength=n_items).float()

    recommended_items = set()
    recommendation_popularities = []

    with torch.no_grad():
        for batch_data in test_data:
            interaction = batch_data[0] if isinstance(batch_data, tuple) else batch_data
            history_index = batch_data[1] if isinstance(batch_data, tuple) and len(batch_data) > 1 else None

            interaction = interaction.to(config["device"])

            scores = model.full_sort_predict(interaction)
            scores = scores.view(-1, n_items)

            if history_index is not None:
                scores[history_index] = -float("inf")

            _, top_items = torch.topk(scores, k=top_k, dim=1)

            for rec_list in top_items.cpu().tolist():
                recommended_items.update(rec_list)

                for item_id in rec_list:
                    recommendation_popularities.append(
                        float(item_popularity[item_id].item())
                    )

    return {
        f"coverage@{top_k}": len(recommended_items) / n_items if n_items > 0 else 0.0,
        f"avg_recommendation_popularity@{top_k}": (
            sum(recommendation_popularities) / len(recommendation_popularities)
            if recommendation_popularities
            else 0.0
        ),
    }


def run_experiment(
    model_class,
    model_name: str,
    dataset_name: str,
    config_updates: dict,
    device: str,
) -> dict:
    project_root = Path(__file__).resolve().parents[3]

    config_dict = {
        "model": model_class,
        "dataset": dataset_name,
        "data_path": str(project_root / "data" / "recbole"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "epochs": 1,
        "train_batch_size": 2048,
        "eval_batch_size": 2048,
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

    config_dict.update(config_updates)

    config = Config(model=model_class, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    train_start = time.time()
    trainer.fit(train_data, valid_data, saved=False, verbose=False)
    train_runtime_seconds = round(time.time() - train_start, 2)

    eval_start = time.time()
    test_result = trainer.evaluate(test_data, load_best_model=False)
    eval_runtime_seconds = round(time.time() - eval_start, 2)

    if ENABLE_EXTRA_METRICS:
        extra_start = time.time()
        extra_metrics = calculate_extra_metrics(
            model=model,
            test_data=test_data,
            train_data=train_data,
            config=config,
            top_k=10,
        )
        extra_runtime_seconds = round(time.time() - extra_start, 2)
    else:
        extra_metrics = {}
        extra_runtime_seconds = 0.0

    return {
        "model": model_name,
        "dataset": dataset_name,
        "device": device,
        "epochs": config["epochs"],
        "train_batch_size": config["train_batch_size"],
        "eval_batch_size": config["eval_batch_size"],
        "train_runtime_seconds": train_runtime_seconds,
        "eval_runtime_seconds": eval_runtime_seconds,
        "extra_metrics_runtime_seconds": extra_runtime_seconds,
        **config_updates,
        **dict(test_result),
        **extra_metrics,
    }


def append_result(output_file: Path, result: dict) -> None:
    result_df = pd.DataFrame([result])

    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(output_file, index=False)


def run_and_store(
    output_file: Path,
    logger: ExperimentLogger,
    completed_run_ids: set[str],
    model_class,
    model_name: str,
    dataset_name: str,
    config_updates: dict,
    device: str,
) -> None:
    run_id = make_run_id(model_name, dataset_name, config_updates)

    if run_id in completed_run_ids:
        print(f"Skipping completed run: {run_id}")
        return

    start_time = time.time()

    try:
        result = run_experiment(
            model_class=model_class,
            model_name=model_name,
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

        result["run_id"] = run_id
        result["status"] = "success"
        result["error_message"] = ""
        result["runtime_seconds"] = round(time.time() - start_time, 2)
        result["config_json"] = ExperimentLogger.serialize_config(config_updates)

        append_result(output_file, result)
        logger.log_result(result)
        completed_run_ids.add(run_id)

    except Exception as error:
        failed_result = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": device,
            "status": "failed",
            "error_message": str(error),
            "runtime_seconds": round(time.time() - start_time, 2),
            "config_json": ExperimentLogger.serialize_config(config_updates),
            **config_updates,
        }

        append_result(output_file, failed_result)
        logger.log_result(failed_result)

        print(f"Run failed for {model_name} on {dataset_name}: {config_updates}")
        print(f"Error: {error}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "topn_full_tuning_results.csv"
    )

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "topn_full_tuning_experiment_log.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_completed_run_ids(output_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Extra metrics enabled: {ENABLE_EXTRA_METRICS}")
    print(f"Already completed runs: {len(completed_run_ids)}")

    datasets = [
        "movielens_recbole",
        "amazon_recbole",
    ]

    for dataset_name in datasets:
        print(f"\n===== DATASET: {dataset_name} =====")

        run_and_store(
            output_file=output_file,
            logger=logger,
            completed_run_ids=completed_run_ids,
            model_class=MostPopRecBole,
            model_name="MostPop",
            dataset_name=dataset_name,
            config_updates={},
            device=device,
        )

        for window_days in [1, 3, 7, 14, 30, 60, 90, 180]:
            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=RecentPopRecBole,
                model_name="RecentPop",
                dataset_name=dataset_name,
                config_updates={"window_days": window_days},
                device=device,
            )

        for decay_lambda in [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]:
            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=DecayPopRecBole,
                model_name="DecayPop",
                dataset_name=dataset_name,
                config_updates={"decay_lambda": decay_lambda},
                device=device,
            )

        for embedding_size, learning_rate, epochs in product(
            [32, 64, 128],
            [0.001, 0.0005, 0.0001],
            [50],
        ):
            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=BPR,
                model_name="BPR",
                dataset_name=dataset_name,
                config_updates={
                    "embedding_size": embedding_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "train_neg_sample_args": {
                        "distribution": "uniform",
                        "sample_num": 1,
                        "alpha": 1.0,
                        "dynamic": False,
                        "candidate_num": 0,
                    },
                },
                device=device,
            )

    print(f"\nSaved Top-N full tuning results to: {output_file}")
    print(f"Saved Top-N full experiment log to: {log_file}")


if __name__ == "__main__":
    main()
