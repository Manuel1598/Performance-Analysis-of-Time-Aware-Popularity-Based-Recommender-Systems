from pathlib import Path
import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.topn.decaypop_recbole import DecayPopRecBole


# =========================
# Dataset Configuration
# =========================
DATASETS = {
    "movielens": {
        "recbole_name": "movielens_recbole",
        "output_prefix": "movielens",
    },
    "amazon": {
        "recbole_name": "amazon_recbole",
        "output_prefix": "amazon",
    },
}


def run_for_dataset(dataset_key: str, dataset_config: dict) -> None:
    project_root = Path(__file__).resolve().parents[3]

    recbole_dataset_name = dataset_config["recbole_name"]
    output_prefix = dataset_config["output_prefix"]

    config_dict = {
        "model": DecayPopRecBole,
        "dataset": recbole_dataset_name,
        "data_path": str(project_root / "data" / "recbole"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"]
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
        "decay_lambda": 1e-7,
        "seed": 42,
        "reproducibility": True,
        "device": "cpu",
        "show_progress": True,
    }

    print("\n" + "=" * 80)
    print(f"Running DecayPopRecBole on dataset: {dataset_key}")
    print("=" * 80)

    print("Project root:", project_root)
    print("RecBole data path:", project_root / "data" / "recbole")

    # =========================
    # RecBole Pipeline
    # =========================
    print("Creating RecBole config...")
    config = Config(model=DecayPopRecBole, config_dict=config_dict)

    print("Creating RecBole dataset...")
    dataset = create_dataset(config)
    print(dataset)

    print("Preparing train/valid/test data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print("Initializing DecayPopRecBole model...")
    model = DecayPopRecBole(config, dataset).to(config["device"])
    print(model)

    print("Creating trainer...")
    trainer = Trainer(config, model)

    print("Running training...")
    trainer.fit(train_data, valid_data, saved=False)

    print("Running evaluation on test set...")
    test_result = trainer.evaluate(test_data, load_best_model=False)

    print("\nTest Results:")
    print(test_result)

    # =========================
    # Save Results
    # =========================
    results_df = pd.DataFrame([test_result])

    output_file = (
        project_root
        / "recbole_results"
        / f"{output_prefix}_decaypop_recbole_metrics.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"Saved RecBole results to: {output_file}")


def main() -> None:
    for dataset_key, dataset_config in DATASETS.items():
        run_for_dataset(dataset_key, dataset_config)


if __name__ == "__main__":
    main()