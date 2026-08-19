import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.merge_validation_first_results import (
    DEFAULT_PROTOCOL_VERSION,
    merge_worker_files,
    write_atomically,
)


def result(run_id: str, metric: str = "0.1") -> dict[str, str]:
    return {
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "run_id": run_id,
        "scenario": "session",
        "dataset": "example",
        "model": "MostPop",
        "status": "success",
        "valid_mrr@10": metric,
    }


class MergeValidationFirstResultsTests(unittest.TestCase):
    def write(self, path: Path, rows: list[dict[str, str]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_merges_disjoint_workers_and_keeps_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write(first, [result("one")])
            self.write(second, [result("two")])

            merged = merge_worker_files([first, second], DEFAULT_PROTOCOL_VERSION)

            self.assertEqual(merged["run_id"].tolist(), ["one", "two"])

    def test_accepts_an_exact_duplicate_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write(first, [result("same")])
            self.write(second, [result("same")])

            merged = merge_worker_files([first, second], DEFAULT_PROTOCOL_VERSION)

            self.assertEqual(merged["run_id"].tolist(), ["same"])

    def test_rejects_a_conflicting_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write(first, [result("same", "0.1")])
            self.write(second, [result("same", "0.2")])

            with self.assertRaisesRegex(ValueError, "Conflicting duplicate run_id"):
                merge_worker_files([first, second], DEFAULT_PROTOCOL_VERSION)

    def test_preserves_other_protocol_rows_when_target_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            legacy = result("legacy")
            legacy["protocol_version"] = "old_protocol"
            self.write(first, [result("one"), legacy])
            self.write(second, [result("two")])

            merged = merge_worker_files([first, second], DEFAULT_PROTOCOL_VERSION)

            self.assertEqual(
                set(merged["protocol_version"]),
                {DEFAULT_PROTOCOL_VERSION, "old_protocol"},
            )

    def test_rejects_an_input_without_required_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            wrong = result("wrong")
            wrong["protocol_version"] = "old_protocol"
            self.write(first, [result("one")])
            self.write(second, [wrong])

            with self.assertRaisesRegex(
                ValueError, "does not contain required protocol version"
            ):
                merge_worker_files([first, second], DEFAULT_PROTOCOL_VERSION)

    def test_atomic_writer_can_replace_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "merged.csv"
            output.write_text("old\n", encoding="utf-8")
            frame = pd.DataFrame([result("one")])

            write_atomically(frame, output)

            self.assertEqual(pd.read_csv(output)["run_id"].tolist(), ["one"])


if __name__ == "__main__":
    unittest.main()
