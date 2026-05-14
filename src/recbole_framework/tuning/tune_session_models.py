from itertools import product
from pathlib import Path

import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole


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
            "inter": ["user_id", "item_id", "timestamp"]
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

    trainer.fit(train_data, valid_data, saved=False, verbose=False)
    test_result = trainer.evaluate(test_data, load_best_model=False)

    return {
        "model": model_name,
        "dataset": dataset_name,
        **config_updates,
        **dict(test_result),
    }


def run_and_store(
    all_results: list[dict],
    output_file: Path,
    model_class,
    model_name: str,
    dataset_name: str,
    config_updates: dict,
    device: str,
) -> None:
    try:
        result = run_experiment(
            model_class=model_class,
            model_name=model_name,
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

        all_results.append(result)
        pd.DataFrame(all_results).to_csv(output_file, index=False)

    except Exception as error:
        print(f"Run failed for {model_name} with config {config_updates}")
        print(f"Error: {error}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "yoochoose_session_tuning_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = []

    dataset_name = "yoochoose_recbole_sample"

    # VS-KNN tuning
    for k, sample_size in product([50, 100, 200], [250, 500, 1000]):
        config_updates = {
            "vsknn_k": k,
            "vsknn_sample_size": sample_size,
        }

        print(f"Running VS-KNN: {config_updates}")

        run_and_store(
            all_results=all_results,
            output_file=output_file,
            model_class=VSKNNRecBole,
            model_name="VS-KNN",
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

    # VSTAN tuning
    for k, sample_size, position_decay, idf_weighting in product(
        [50, 100, 200],
        [250, 500],
        [0.05, 0.1, 0.2],
        [True, False],
    ):
        config_updates = {
            "vstan_k": k,
            "vstan_sample_size": sample_size,
            "vstan_position_decay": position_decay,
            "vstan_idf_weighting": idf_weighting,
        }

        print(f"Running VSTAN: {config_updates}")

        run_and_store(
            all_results=all_results,
            output_file=output_file,
            model_class=VSTANRecBole,
            model_name="VSTAN",
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

    # GRU4Rec tuning
    for hidden_size, learning_rate, dropout_prob, epochs in product(
        [64, 128],
        [0.001, 0.0005],
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

        print(f"Running GRU4Rec: {config_updates}")

        run_and_store(
            all_results=all_results,
            output_file=output_file,
            model_class=GRU4Rec,
            model_name="GRU4Rec",
            dataset_name=dataset_name,
            config_updates=config_updates,
            device=device,
        )

    results_df = pd.DataFrame(all_results)

    print("\nBest configurations by MRR@10:")
    print(results_df.sort_values("mrr@10", ascending=False).head(10))

    print(f"\nSaved tuning results to: {output_file}")


if __name__ == "__main__":
    main()