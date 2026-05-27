from itertools import product
from pathlib import Path
import time

import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger


ENABLE_EXTRA_METRICS = False


def extract_interaction(batch_data):
    if isinstance(batch_data, tuple):
        return batch_data[0]
    return batch_data


def extract_history_index(batch_data):
    if isinstance(batch_data, tuple) and len(batch_data) > 1:
        return batch_data[1]
    return None


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


def calculate_extra_metrics(
    model,
    test_data,
    train_data,
    config,
    top_k: int = 10,
) -> dict:
    model.eval()

    item_field = config["ITEM_ID_FIELD"]
    n_items = train_data.dataset.num(item_field)

    train_item_ids = train_data.dataset.inter_feat[item_field].long().cpu()
    item_popularity = torch.bincount(train_item_ids, minlength=n_items).float()

    recommended_items = set()
    recommendation_popularities = []

    with torch.no_grad():
        for batch_data in test_data:
            interaction = extract_interaction(batch_data)
            history_index = extract_history_index(batch_data)

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

    coverage = len(recommended_items) / n_items if n_items > 0 else 0.0

    avg_popularity = (
        sum(recommendation_popularities) / len(recommendation_popularities)
        if recommendation_popularities
        else 0.0
    )

    return {
        f"coverage@{top_k}": coverage,
        f"avg_recommendation_popularity@{top_k}": avg_popularity,
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
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"],
        },
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
        extra_metrics_start = time.time()

        extra_metrics = calculate_extra_metrics(
            model=model,
            test_data=test_data,
            train_data=train_data,
            config=config,
            top_k=10,
        )

        extra_metrics_runtime_seconds = round(
            time.time() - extra_metrics_start,
            2,
        )
    else:
        extra_metrics = {}
        extra_metrics_runtime_seconds = 0.0

    return {
        "model": model_name,
        "dataset": dataset_name,
        "device": device,
        "epochs": config["epochs"],
        "train_batch_size": config["train_batch_size"],
        "eval_batch_size": config["eval_batch_size"],
        "train_runtime_seconds": train_runtime_seconds,
        "eval_runtime_seconds": eval_runtime_seconds,
        "extra_metrics_runtime_seconds": extra_metrics_runtime_seconds,
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

        runtime_seconds = round(time.time() - start_time, 2)

        result["run_id"] = run_id
        result["status"] = "success"
        result["error_message"] = ""
        result["runtime_seconds"] = runtime_seconds
        result["config_json"] = ExperimentLogger.serialize_config(config_updates)

        append_result(output_file, result)
        logger.log_result(result)

        completed_run_ids.add(run_id)

    except Exception as error:
        runtime_seconds = round(time.time() - start_time, 2)

        failed_result = {
            "run_id": run_id,
            "model": model_name,
            "dataset": dataset_name,
            "device": device,
            "status": "failed",
            "error_message": str(error),
            "runtime_seconds": runtime_seconds,
            "config_json": ExperimentLogger.serialize_config(config_updates),
            **config_updates,
        }

        append_result(output_file, failed_result)
        logger.log_result(failed_result)

        print(
            f"Run failed for {model_name} on {dataset_name} "
            f"with config {config_updates}"
        )
        print(f"Error: {error}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "session_full_tuning_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "session_full_tuning_experiment_log.csv"
    )

    logger = ExperimentLogger(log_file)

    completed_run_ids = load_completed_run_ids(output_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Extra metrics enabled: {ENABLE_EXTRA_METRICS}")
    print(f"Already completed runs: {len(completed_run_ids)}")

    datasets = [
        "yoochoose_recbole_sample",
        "globo_recbole_sample",
        "adressa_recbole_sample",
    ]

    for dataset_name in datasets:
        print(f"\n===== DATASET: {dataset_name} =====")

        for k, sample_size in product(
            [100, 200, 500],
            [100, 250, 500],
        ):
            config_updates = {
                "vsknn_k": k,
                "vsknn_sample_size": sample_size,
            }

            print(f"Running VS-KNN on {dataset_name}: {config_updates}")

            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=VSKNNRecBole,
                model_name="VS-KNN",
                dataset_name=dataset_name,
                config_updates=config_updates,
                device=device,
            )

        for k, sample_size, position_decay, idf_weighting in product(
            [100, 200, 500],
            [100, 250, 500],
            [0.05, 0.1, 0.2],
            [True, False],
        ):
            config_updates = {
                "vstan_k": k,
                "vstan_sample_size": sample_size,
                "vstan_position_decay": position_decay,
                "vstan_idf_weighting": idf_weighting,
            }

            print(f"Running VSTAN on {dataset_name}: {config_updates}")

            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=VSTANRecBole,
                model_name="VSTAN",
                dataset_name=dataset_name,
                config_updates=config_updates,
                device=device,
            )

        for hidden_size, learning_rate, dropout_prob, epochs in product(
            [100, 128, 256],
            [0.001, 0.0005, 0.0001],
            [0.1, 0.2],
            [10, 20],
        ):
            config_updates = {
                "model": "GRU4Rec",
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "dropout_prob": dropout_prob,
                "epochs": epochs,
                "num_layers": 1,
                "loss_type": "CE",
                "train_neg_sample_args": None,
                "train_batch_size": 2048,
                "eval_batch_size": 2048,
            }

            print(f"Running GRU4Rec on {dataset_name}: {config_updates}")

            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=GRU4Rec,
                model_name="GRU4Rec",
                dataset_name=dataset_name,
                config_updates=config_updates,
                device=device,
            )

    print(f"\nSaved full tuning results to: {output_file}")
    print(f"Saved full experiment log to: {log_file}")


if __name__ == "__main__":
    main()
