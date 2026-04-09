import pandas as pd

from src.models.base import BaseRecommender


class RecentPopRecommender(BaseRecommender):
    def __init__(self, window_days: int = 30) -> None:
        self.window_days = window_days
        self.train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df

    def compute_recent_popularity(
        self,
        reference_timestamp: int
    ) -> pd.DataFrame:
        if self.train_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        print(f"Computing RecentPop popularity for reference time {reference_timestamp}...")

        window_seconds = self.window_days * 24 * 60 * 60
        window_start = reference_timestamp - window_seconds

        recent_df = self.train_df[
            (self.train_df["timestamp"] >= window_start) &
            (self.train_df["timestamp"] <= reference_timestamp)
        ].copy()

        popularity_df = (
            recent_df.groupby("item_id")
            .size()
            .reset_index(name="interaction_count")
            .sort_values(by="interaction_count", ascending=False)
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
            raise ValueError("RecentPop requires a reference_timestamp.")

        popularity_df = self.compute_recent_popularity(reference_timestamp)

        seen_items = user_seen.get(user_id, set())
        recommendations = []

        for item_id in popularity_df["item_id"]:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == top_k:
                break

        return recommendations