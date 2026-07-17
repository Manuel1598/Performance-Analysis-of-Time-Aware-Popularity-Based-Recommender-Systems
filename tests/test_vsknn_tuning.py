import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.recbole_framework.tuning.tune_vsknn_audited import (
    BASELINE_CONFIG,
    make_run_id,
    save_result,
    select_best,
    tuning_configs,
)


class VSKNNTuningTests(unittest.TestCase):
    def test_compact_grid_has_unique_baseline_plus_eight_variants(self):
        configs = tuning_configs()
        self.assertEqual(len(configs), 9)
        self.assertEqual(configs[0], BASELINE_CONFIG)
        self.assertEqual(len({make_run_id("dataset", c) for c in configs}), 9)

    def test_run_id_is_stable_across_dictionary_order(self):
        reversed_config = dict(reversed(list(BASELINE_CONFIG.items())))
        self.assertEqual(
            make_run_id("dataset", BASELINE_CONFIG),
            make_run_id("dataset", reversed_config),
        )

    def test_best_selection_uses_mrr_per_dataset_and_ignores_failures(self):
        rows = pd.DataFrame(
            [
                {"dataset": "a", "status": "success", "mrr@10": 0.1},
                {"dataset": "a", "status": "success", "mrr@10": 0.2},
                {"dataset": "b", "status": "failed", "mrr@10": 0.9},
                {"dataset": "b", "status": "success", "mrr@10": 0.3},
            ]
        )
        best = select_best(rows)
        self.assertEqual(dict(zip(best["dataset"], best["mrr@10"])), {"a": 0.2, "b": 0.3})

    def test_save_result_appends_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results.csv"
            save_result(output, {"run_id": "one", "status": "success"})
            save_result(output, {"run_id": "two", "status": "success"})
            self.assertEqual(pd.read_csv(output)["run_id"].tolist(), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
