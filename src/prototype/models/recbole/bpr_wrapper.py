from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer

from src.prototype.models.base import BaseRecommender


class RecBoleBPRRecommender(BaseRecommender):
    def __init__(
        self,
        dataset_name: str = "movielens_recbole",
        data_parent_path: Path | None = None,
        epochs: int = 10,
        train_batch_size: int = 2048,
        eval_batch_size: int = 2048,
        learning_rate: float = 0.001,
        embedding_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.dataset_name = dataset_name
        self.data_parent_path = data_parent_path
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.device = device

        self.config = None
        self.dataset = None
        self.model = None
        self.trainer = None

    def fit(self, train_df: pd.DataFrame) -> None:
        if self.data_parent_path is None:
            raise ValueError("data_parent_path must be provided.")

        print("Loading RecBole dataset...")

        config_dict = {
            "model": "BPR",
            "dataset": self.dataset_name,
            "data_path": str(self.data_parent_path),
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "TIME_FIELD": "timestamp",
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp"]
            },
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "embedding_size": self.embedding_size,
            "device": self.device,
            "checkpoint_dir": "saved",
            "show_progress": True,
        }

        self.config = Config(model=BPR, config_dict=config_dict)

        self.dataset = create_dataset(self.config)
        train_data, valid_data, test_data = data_preparation(self.config, self.dataset)

        print("Training RecBole BPR model...")
        self.model = BPR(self.config, train_data.dataset).to(self.config["device"])
        self.trainer = Trainer(self.config, self.model)
        self.trainer.fit(train_data, valid_data, verbose=True)

    def recommend(
        self,
        user_id: int,
        user_seen: dict[int, set[int]],
        top_k: int = 10,
        reference_timestamp: int | None = None,
    ) -> list[int]:
        if self.model is None or self.dataset is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field

        user_token = str(user_id)

        if user_token not in self.dataset.field2token_id[uid_field]:
            return []

        internal_user_id = self.dataset.token2id(uid_field, user_token)

        interaction = {
            uid_field: torch.tensor([internal_user_id], device=self.config["device"])
        }

        scores = self.model.full_sort_predict(interaction)
        scores = scores.view(-1).detach().cpu().numpy()

        ranked_internal_item_ids = np.argsort(-scores)

        seen_items = user_seen.get(user_id, set())
        recommendations: list[int] = []

        for internal_item_id in ranked_internal_item_ids:
            external_item_token = self.dataset.id2token(iid_field, int(internal_item_id))

            # skip padding / invalid token
            if external_item_token == "[PAD]":
                continue

            external_item_id = int(external_item_token)

            if external_item_id in seen_items:
                continue

            recommendations.append(external_item_id)

            if len(recommendations) == top_k:
                break

        return recommendations