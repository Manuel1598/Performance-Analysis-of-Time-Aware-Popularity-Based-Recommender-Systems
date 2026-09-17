import unittest

import pandas as pd

from src.prototype.datapipeline.preprocessing_amazon import preprocess_amazon


class AmazonPreprocessingTests(unittest.TestCase):
    def test_keeps_only_users_with_two_positive_ratings(self):
        reviews = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
                "item_id": ["a", "b", "c", "d", "e", "f", "g"],
                "rating": [5.0, 3.0, 4.0, 5.0, 2.0, 4.0, 4.5],
                "timestamp": [3, 1, 2, 1, 2, 2, 1],
            }
        )

        result = preprocess_amazon(reviews)

        self.assertEqual(
            result.to_dict("records"),
            [
                {"user_id": "u1", "item_id": "c", "timestamp": 2},
                {"user_id": "u1", "item_id": "a", "timestamp": 3},
                {"user_id": "u3", "item_id": "g", "timestamp": 1},
                {"user_id": "u3", "item_id": "f", "timestamp": 2},
            ],
        )

    def test_rating_threshold_is_inclusive(self):
        reviews = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["a", "b", "c"],
                "rating": [3.999, 4.0, 5.0],
                "timestamp": [1, 2, 3],
            }
        )

        result = preprocess_amazon(reviews)

        self.assertEqual(result["item_id"].tolist(), ["b", "c"])

    def test_rating_column_is_required(self):
        reviews = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["a", "b"],
                "timestamp": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "rating"):
            preprocess_amazon(reviews)

    def test_minimum_user_history_cannot_be_less_than_two(self):
        reviews = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["a", "b"],
                "rating": [4.0, 5.0],
                "timestamp": [1, 2],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least 2"):
            preprocess_amazon(reviews, min_interactions_per_user=1)


if __name__ == "__main__":
    unittest.main()
