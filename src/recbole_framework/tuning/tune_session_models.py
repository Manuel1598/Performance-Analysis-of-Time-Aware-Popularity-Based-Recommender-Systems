from itertools import product
from pathlib import Path
import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole


def run_experiment(model_class, model_name: str, dataset_name: str, config_updates: dict) -> dict:
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
        "device": "cpu",
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

    result = {
        "model": model_name,
        "dataset": dataset_name,
        **config_updates,
        **dict(test_result),
    }

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "yoochoose_session_tuning_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # VS-KNN tuning
    for k, sample_size in product([50, 100, 200], [250, 500, 1000]):
        print(f"Running VS-KNN: k={k}, sample_size={sample_size}")

        result = run_experiment(
            model_class=VSKNNRecBole,
            model_name="VS-KNN",
            dataset_name="yoochoose_recbole_sample",
            config_updates={
                "vsknn_k": k,
                "vsknn_sample_size": sample_size,
            },
        )

        all_results.append(result)
        pd.DataFrame(all_results).to_csv(output_file, index=False)

    # VSTAN tuning
    for k, sample_size, position_decay, idf_weighting in product(
        [50, 100, 200],
        [250, 500],
        [0.05, 0.1, 0.2],
        [True, False],
    ):
        print(
            f"Running VSTAN: k={k}, sample_size={sample_size}, "
            f"position_decay={position_decay}, idf_weighting={idf_weighting}"
        )

        result = run_experiment(
            model_class=VSTANRecBole,
            model_name="VSTAN",
            dataset_name="yoochoose_recbole_sample",
            config_updates={
                "vstan_k": k,
                "vstan_sample_size": sample_size,
                "vstan_position_decay": position_decay,
                "vstan_idf_weighting": idf_weighting,
            },
        )

        all_results.append(result)
        pd.DataFrame(all_results).to_csv(output_file, index=False)

    results_df = pd.DataFrame(all_results)

    print("\nBest configurations by MRR@10:")
    print(results_df.sort_values("mrr@10", ascending=False).head(10))

    print(f"\nSaved tuning results to: {output_file}")


if __name__ == "__main__":
    main()