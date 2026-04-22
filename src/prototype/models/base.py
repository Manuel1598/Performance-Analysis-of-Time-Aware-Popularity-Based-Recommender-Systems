from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the recommender on the training data."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        user_seen: dict[int, set[int]],
        top_k: int = 10,
        reference_timestamp: int | None = None,
    ) -> list[int]:
        """Generate a top-k recommendation list for one user."""
        pass