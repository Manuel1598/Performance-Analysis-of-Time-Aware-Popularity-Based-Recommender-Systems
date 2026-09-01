import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.consolidate_validation_first_v6 import (
    consolidate_rows,
    expected_final_rows,
    expected_validation_rows,
    expected_optional_final_rows,
    read_sources,
    select_validation_winners,
    SourceSpec,
    VALIDATION_REQUIRED_COLUMNS,
    VALIDATION_TOLERATED_SUCCESS_DIFFERENCES,
)
from tools.run_validation_first_experiments import (
    build_final_test_summary,
)



PROTOCOL = "validation_first_v6"


def validation_row(
    run_id: str,
    status: str,
    metric: str = "0.1",
    runtime: str = "10",
    error: str = "",
    config_json: str = "{}",
) -> dict[str, str]:
    return {
        "protocol_version": PROTOCOL,
        "evaluated_split": "validation",
        "run_id": run_id,
        "scenario": "topn",
        "dataset": "amazon_recbole",
        "model": "BPR",
        "device": "cpu",
        "best_valid_score": metric if status == "success" else "",
        "valid_mrr@10": metric if status == "success" else "",
        "runtime_seconds": runtime,
        "config_json": config_json,
        "status": status,
        "error": error,
    }


def source_frame(rows: list[dict[str, str]], order: int, label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["_source_order"] = order
    frame["_source_label"] = label
    frame["_source_path"] = f"{label}/validation_trials.csv"
    frame["_source_row"] = range(2, len(frame) + 2)
    return frame


class ConsolidateValidationFirstV6Tests(unittest.TestCase):
    def test_expected_protocol_counts_are_stable(self):
        self.assertEqual(len(expected_validation_rows()), 197)
        self.assertEqual(len(expected_final_rows()), 26)
        self.assertEqual(len(expected_optional_final_rows()), 5)

    def test_successful_retry_replaces_failed_row_with_same_id(self):
        failed = source_frame(
            [validation_row("same", "failed", error="old dataset path")],
            0,
            "pc2",
        )
        successful = source_frame(
            [validation_row("same", "success", metric="0.04", runtime="99")],
            1,
            "pc1",
        )

        merged = consolidate_rows(
            pd.concat([failed, successful], ignore_index=True),
            "run_id",
            VALIDATION_TOLERATED_SUCCESS_DIFFERENCES,
        )

        self.assertEqual(merged.frame.iloc[0]["status"], "success")
        self.assertEqual(
            merged.decisions.iloc[0]["decision"], "success_replaces_failed"
        )
        self.assertEqual(merged.decisions.iloc[0]["selected_source_label"], "pc1")

    def test_equal_named_files_in_different_directories_are_distinct_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first" / "validation_trials.csv"
            second = root / "second" / "validation_trials.csv"
            first.parent.mkdir()
            second.parent.mkdir()
            pd.DataFrame([validation_row("one", "success")]).to_csv(first, index=False)
            pd.DataFrame([validation_row("two", "success")]).to_csv(second, index=False)

            frame, inventory = read_sources(
                [SourceSpec("first", first), SourceSpec("second", second)],
                PROTOCOL,
                VALIDATION_REQUIRED_COLUMNS,
                "run_id",
                False,
            )

            self.assertEqual(set(frame["run_id"]), {"one", "two"})
            self.assertEqual(inventory["file_name"].tolist(), [
                "validation_trials.csv",
                "validation_trials.csv",
            ])
            self.assertNotEqual(
                inventory.iloc[0]["source_path"], inventory.iloc[1]["source_path"]
            )

    def test_success_duplicates_may_differ_only_in_validation_runtime(self):
        first = source_frame(
            [validation_row("same", "success", metric="0.04", runtime="100")],
            0,
            "pc2",
        )
        second = source_frame(
            [validation_row("same", "success", metric="0.04", runtime="90")],
            1,
            "pc1",
        )

        merged = consolidate_rows(
            pd.concat([first, second], ignore_index=True),
            "run_id",
            VALIDATION_TOLERATED_SUCCESS_DIFFERENCES,
        )

        self.assertEqual(merged.frame.iloc[0]["runtime_seconds"], "100")
        self.assertEqual(
            merged.decisions.iloc[0]["decision"],
            "equivalent_success_prefer_first_source",
        )

    def test_conflicting_success_metrics_are_rejected(self):
        first = source_frame(
            [validation_row("same", "success", metric="0.04")], 0, "pc2"
        )
        second = source_frame(
            [validation_row("same", "success", metric="0.05")], 1, "pc1"
        )

        with self.assertRaisesRegex(ValueError, "Conflicting successful run_id"):
            consolidate_rows(
                pd.concat([first, second], ignore_index=True),
                "run_id",
                VALIDATION_TOLERATED_SUCCESS_DIFFERENCES,
            )

    def test_winner_order_matches_runner(self):
        rows = pd.DataFrame(
            [
                validation_row("slow", "success", metric="0.2", runtime="20"),
                validation_row("lower", "success", metric="0.1", runtime="1"),
                validation_row("fast", "success", metric="0.2", runtime="10"),
            ]
        )

        selected = select_validation_winners(rows)

        self.assertEqual(selected.iloc[0]["run_id"], "fast")


    def test_resource_excluded_bpr_trial_cannot_win(self):
        rows = pd.DataFrame(
            [
                validation_row(
                    "eligible",
                    "success",
                    metric="0.2",
                    config_json='{"embedding_size":256}',
                ),
                validation_row(
                    "excluded",
                    "success",
                    metric="0.9",
                    config_json='{"embedding_size":512}',
                ),
            ]
        )

        selected = select_validation_winners(rows)

        self.assertEqual(selected.iloc[0]["run_id"], "eligible")
        self.assertTrue(bool(selected.iloc[0]["selection_eligible"]))

    def test_optional_seed_does_not_change_primary_summary(self):
        frame = pd.DataFrame(
            [
                {
                    "protocol_version": PROTOCOL,
                    "scenario": "topn",
                    "dataset": "amazon_recbole",
                    "model": "BPR",
                    "device": "cpu",
                    "seed": 42,
                    "mrr@10": 0.1,
                    "runtime_seconds": 10,
                    "config_json": "{}",
                },
                {
                    "protocol_version": PROTOCOL,
                    "scenario": "topn",
                    "dataset": "amazon_recbole",
                    "model": "BPR",
                    "device": "cpu",
                    "seed": 43,
                    "mrr@10": 0.2,
                    "runtime_seconds": 20,
                    "config_json": "{}",
                },
            ]
        )

        summary = build_final_test_summary(frame).iloc[0]

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["expected_seed_count"], 1)
        self.assertEqual(summary["optional_seed_count"], 1)
        self.assertAlmostEqual(summary["mrr@10_mean"], 0.1)
        self.assertAlmostEqual(summary["mrr@10_optional_mean"], 0.2)
        self.assertAlmostEqual(
            summary["mrr@10_optional_difference_from_primary"], 0.1
        )

if __name__ == "__main__":
    unittest.main()
