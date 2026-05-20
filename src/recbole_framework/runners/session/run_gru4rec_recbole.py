from pathlib import Path
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_dict = {
        "model": "GRU4Rec",
        "dataset": "yoochoose_recbole_sample",
        "data_path": str(project_root / "data" / "recbole"),

        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"]
        },

        "MAX_ITEM_LIST_LENGTH": 20,

        "epochs": 20,
        "train_batch_size": 2048,
        "eval_batch_size": 2048,
        "learning_rate": 0.001,

        "hidden_size": 128,
        "num_layers": 1,
        "dropout_prob": 0.2,
        "loss_type": "CE",
        "train_neg_sample_args": None,

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
        "show_progress": True,
    }

    print("Creating RecBole config...")
    config = Config(model=GRU4Rec, config_dict=config_dict)

    print("Creating RecBole dataset...")
    dataset = create_dataset(config)
    print(dataset)

    print("Preparing train/valid/test data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print("Initializing GRU4Rec model...")
    model = GRU4Rec(config, train_data.dataset).to(config["device"])
    print(model)

    trainer = Trainer(config, model)

    print("Running training...")
    trainer.fit(train_data, valid_data, saved=False)

    print("Running evaluation on test set...")
    test_result = trainer.evaluate(test_data, load_best_model=False)

    print("\nTest Results:")
    print(test_result)

    output_file = (
        project_root
        / "recbole_results"
        / "yoochoose_sample_gru4rec_recbole_metrics.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([test_result]).to_csv(output_file, index=False)

    print(f"Saved RecBole results to: {output_file}")


if __name__ == "__main__":
    main()