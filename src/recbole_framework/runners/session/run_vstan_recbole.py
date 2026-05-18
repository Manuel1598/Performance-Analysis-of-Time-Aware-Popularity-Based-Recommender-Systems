from pathlib import Path
import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    config_dict = {
        "model": VSTANRecBole,
        "dataset": "yoochoose_recbole_sample",
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
        "vstan_k": 100,
        "vstan_sample_size": 500,
        "vstan_position_decay": 0.1,
        "vstan_idf_weighting": True,
        "seed": 42,
        "reproducibility": True,
        "device": "cpu",
        "show_progress": True,
    }

    print("Project root:", project_root)
    print("RecBole data path:", project_root / "data" / "recbole")

    print("Creating RecBole config...")
    config = Config(model=VSTANRecBole, config_dict=config_dict)

    print("Creating RecBole dataset...")
    dataset = create_dataset(config)
    print(dataset)

    print("Preparing train/valid/test data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print("Initializing VSTANRecBole model...")
    model = VSTANRecBole(config, train_data.dataset).to(config["device"])
    print(model)

    print("Creating trainer...")
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
        / "yoochoose_sample_vstan_recbole_metrics.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([test_result]).to_csv(output_file, index=False)

    print(f"Saved RecBole results to: {output_file}")


if __name__ == "__main__":
    main()