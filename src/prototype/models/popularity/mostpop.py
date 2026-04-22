import pandas as pd

from src.prototype.models.base import BaseRecommender


class MostPopRecommender(BaseRecommender):
    def __init__(self) -> None:
        self.popularity_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        print("Computing item popularity from training interactions...")

        self.popularity_df = (
            train_df.groupby("item_id")
            .size()
            .reset_index(name="interaction_count")
            .sort_values(by="interaction_count", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(
        self,
        user_id: int,
        user_seen: dict[int, set[int]],
        top_k: int = 10,
        reference_timestamp: int | None = None,
    ) -> list[int]:
        if self.popularity_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        seen_items = user_seen.get(user_id, set())
        recommendations = []

        for item_id in self.popularity_df["item_id"]:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == top_k:
                break

        return recommendations