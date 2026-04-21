from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.models.recbole.custom_models.mostpop_recbole import MostPopRecBole


def main():
    config_dict = {
        "model": "MostPopRecBole",
        "dataset": "movielens_recbole",
        "data_path": "data/recbole",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "timestamp"]
        },
        "epochs": 1,
        "topk": [10],
        "metrics": ["Hit", "NDCG", "MRR"],
        "valid_metric": "MRR@10",
    }

    config = Config(model=MostPopRecBole, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MostPopRecBole(config, dataset)
    trainer = Trainer(config, model)

    trainer.fit(train_data, valid_data)
    result = trainer.evaluate(test_data)

    print(result)


if __name__ == "__main__":
    main()