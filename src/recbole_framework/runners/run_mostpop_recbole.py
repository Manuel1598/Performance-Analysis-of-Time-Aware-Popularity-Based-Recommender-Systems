from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.mostpop_recbole import MostPopRecBole


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    config_dict = {
        "model": MostPopRecBole,
        "dataset": "movielens_recbole",
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
            "mode": "full"
        },
        "seed": 42,
        "reproducibility": True,
        "device": "cpu",
        "show_progress": True,
    }

    print("Creating RecBole config...")
    config = Config(model=MostPopRecBole, config_dict=config_dict)

    print("Creating RecBole dataset...")
    dataset = create_dataset(config)
    print(dataset)

    print("Preparing train/valid/test data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print("Initializing MostPopRecBole model...")
    model = MostPopRecBole(config, dataset).to(config["device"])
    print(model)

    print("Creating trainer...")
    trainer = Trainer(config, model)

    print("Running training...")
    trainer.fit(train_data, valid_data, saved=False)

    print("Running evaluation on test set...")
    test_result = trainer.evaluate(test_data, load_best_model=False)

    print("\nTest Results:")
    print(test_result)


if __name__ == "__main__":
    main()