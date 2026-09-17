import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.prototype.datapipeline.preprocessing_movielens import (
    prepare_movielens_interactions,
    resolve_movielens_input,
)


class MovieLensPreprocessingTests(unittest.TestCase):
    def test_keeps_only_users_with_two_positive_ratings(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 3, 3],
                "movieId": [10, 11, 12, 20, 21, 30, 31],
                "rating": [5.0, 3.5, 4.0, 5.0, 2.0, 4.0, 4.5],
                "timestamp": [3, 1, 2, 1, 2, 2, 1],
            }
        )

        result = prepare_movielens_interactions(ratings)

        self.assertEqual(
            result.to_dict("records"),
            [
                {"user_id": 1, "item_id": 12, "timestamp": 2},
                {"user_id": 1, "item_id": 10, "timestamp": 3},
                {"user_id": 3, "item_id": 31, "timestamp": 1},
                {"user_id": 3, "item_id": 30, "timestamp": 2},
            ],
        )

    def test_rating_threshold_is_inclusive(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1],
                "movieId": [10, 11, 12],
                "rating": [3.999, 4.0, 5.0],
                "timestamp": [1, 2, 3],
            }
        )

        result = prepare_movielens_interactions(ratings)

        self.assertEqual(result["item_id"].tolist(), [11, 12])

    def test_rating_column_is_required(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1],
                "movieId": [10, 11],
                "timestamp": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "rating"):
            prepare_movielens_interactions(ratings)

    def test_minimum_user_history_cannot_be_less_than_two(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1],
                "movieId": [10, 11],
                "rating": [4.0, 5.0],
                "timestamp": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least 2"):
            prepare_movielens_interactions(
                ratings, min_interactions_per_user=1
            )

    def test_finds_direct_or_nested_ratings_file(self):
        with TemporaryDirectory() as temporary_directory:
            raw_directory = Path(temporary_directory)
            direct_file = raw_directory / "ratings.csv"
            direct_file.touch()
            self.assertEqual(resolve_movielens_input(raw_directory), direct_file)

            direct_file.unlink()
            nested_file = raw_directory / "ml-20m" / "ratings.csv"
            nested_file.parent.mkdir()
            nested_file.touch()
            self.assertEqual(resolve_movielens_input(raw_directory), nested_file)


if __name__ == "__main__":
    unittest.main()
