import numpy as np
import pandas as pd

from src.models.base import BaseRecommender


class DecayPopRecommender(BaseRecommender):
    def __init__(self, decay_lambda: float = 1e-7) -> None:
        self.decay_lambda = decay_lambda
        self.train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df

    def compute_decay_popularity(
        self,
        reference_timestamp: int
    ) -> pd.DataFrame:
        if self.train_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        print(f"Computing DecayPop popularity for reference time {reference_timestamp}...")

        df = self.train_df[self.train_df["timestamp"] <= reference_timestamp].copy()

        df["time_diff"] = reference_timestamp - df["timestamp"]
        df["weight"] = np.exp(-self.decay_lambda * df["time_diff"])

        popularity_df = (
            df.groupby("item_id")["weight"]
            .sum()
            .reset_index()
            .sort_values(by="weight", ascending=False)
            .reset_index(drop=True)
        )

        return popularity_df

    def recommend(
        self,
        user_id: int,
        user_seen: dict[int, set[int]],
        top_k: int = 10,
        reference_timestamp: int | None = None,
    ) -> list[int]:
        if reference_timestamp is None:
            raise ValueError("DecayPop requires a reference_timestamp.")

        popularity_df = self.compute_decay_popularity(reference_timestamp)

        seen_items = user_seen.get(user_id, set())
        recommendations = []

        for item_id in popularity_df["item_id"]:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == top_k:
                break

        return recommendations