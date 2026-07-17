import unittest

import pandas as pd

from src.recbole_framework.analysis.build_best_model_overview import (
    clean_successful_results,
    select_best_per_model_dataset,
)


class BestModelOverviewTests(unittest.TestCase):
    def test_cleaner_removes_failures_headers_and_invalid_mrr(self):
        data = pd.DataFrame(
            [
                {"model": "A", "dataset": "d", "status": "success", "mrr@10": "0.2"},
                {"model": "B", "dataset": "d", "status": "failed", "mrr@10": "0.9"},
                {"model": "model", "dataset": "dataset", "status": "status", "mrr@10": "mrr@10"},
                {"model": "C", "dataset": "d", "status": "success", "mrr@10": "bad"},
            ]
        )
        cleaned = clean_successful_results(data)
        self.assertEqual(cleaned["model"].tolist(), ["A"])
        self.assertEqual(cleaned["mrr@10"].tolist(), [0.2])

    def test_selection_prefers_mrr_then_lower_runtime(self):
        data = pd.DataFrame(
            [
                {"model": "A", "dataset": "d", "mrr@10": 0.2, "runtime_seconds": 20},
                {"model": "A", "dataset": "d", "mrr@10": 0.3, "runtime_seconds": 30},
                {"model": "A", "dataset": "d", "mrr@10": 0.3, "runtime_seconds": 10},
            ]
        )
        best = select_best_per_model_dataset(data)
        self.assertEqual(best.iloc[0]["runtime_seconds"], 10)


if __name__ == "__main__":
    unittest.main()
