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
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"],
        },
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

    return {
        "model": model_name,
        "dataset": dataset_name,
        "device": device,
        "epochs": config["epochs"],
        "train_batch_size": config["train_batch_size"],
        "eval_batch_size": config["eval_batch_size"],
        "train_runtime_seconds": train_runtime_seconds,
        "eval_runtime_seconds": eval_runtime_seconds,
        "extra_metrics_runtime_seconds": 0.0,
        **config_updates,
        **dict(test_result),
    }


def run_and_store(
    all_results: list[dict],
    output_file: Path,
    logger: ExperimentLogger,
    model_class,
    model_name: str,
    dataset_name: str,
    config_updates: dict,
    device: str,
) -> None:
    start_time = time.time()

    try:
        result = run_experiment(
            model_class=model_class,
            model_name=model_name,
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

        result["status"] = "success"
        result["error_message"] = ""
        result["runtime_seconds"] = round(time.time() - start_time, 2)
        result["config_json"] = ExperimentLogger.serialize_config(config_updates)

        all_results.append(result)
        pd.DataFrame(all_results).to_csv(output_file, index=False)
        logger.log_result(result)

    except Exception as error:
        failed_result = {
            "model": model_name,
            "dataset": dataset_name,
            "device": device,
            "status": "failed",
            "error_message": str(error),
            "runtime_seconds": round(time.time() - start_time, 2),
            "config_json": ExperimentLogger.serialize_config(config_updates),
            **config_updates,
        }

        logger.log_result(failed_result)

        print(f"Run failed for {model_name} on {dataset_name}: {config_updates}")
        print(f"Error: {error}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "topn_tuning_results.csv"
    )

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "topn_tuning_experiment_log.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    all_results = []

    datasets = [
        "movielens_recbole",
        "amazon_recbole",
    ]

    for dataset_name in datasets:
        print(f"\n===== DATASET: {dataset_name} =====")

        run_and_store(
            all_results=all_results,
            output_file=output_file,
            logger=logger,
            model_class=MostPopRecBole,
            model_name="MostPop",
            dataset_name=dataset_name,
            config_updates={},
            device=device,
        )

        for window_days in [7, 30]:
            run_and_store(
                all_results=all_results,
                output_file=output_file,
                logger=logger,
                model_class=RecentPopRecBole,
                model_name="RecentPop",
                dataset_name=dataset_name,
                config_updates={"window_days": window_days},
                device=device,
            )

        for decay_lambda in [1e-7, 1e-6]:
            run_and_store(
                all_results=all_results,
                output_file=output_file,
                logger=logger,
                model_class=DecayPopRecBole,
                model_name="DecayPop",
                dataset_name=dataset_name,
                config_updates={"decay_lambda": decay_lambda},
                device=device,
            )

        for embedding_size, learning_rate, epochs in product(
            [64],
            [0.001],
            [10],
        ):
            run_and_store(
                all_results=all_results,
                output_file=output_file,
                logger=logger,
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

    print(f"\nSaved Top-N tuning results to: {output_file}")
    print(f"Saved Top-N experiment log to: {log_file}")


if __name__ == "__main__":
    main()