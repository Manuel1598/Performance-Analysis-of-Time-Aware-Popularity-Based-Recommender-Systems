import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.run_validation_first_v6_reporting import (
    provisional_source_config,
    source_contains_protocol,
)


class RunValidationFirstV6ReportingTests(unittest.TestCase):
    def test_detects_requested_protocol_without_using_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation_trials.csv"
            pd.DataFrame(
                [
                    {"protocol_version": "old"},
                    {"protocol_version": "validation_first_v6"},
                ]
            ).to_csv(path, index=False)

            self.assertTrue(source_contains_protocol(path, "validation_first_v6"))
            self.assertFalse(source_contains_protocol(path, "missing"))

    def test_provisional_config_omits_final_source_with_only_old_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.csv"
            current = root / "current.csv"
            pd.DataFrame([{"protocol_version": "validation_first_v1"}]).to_csv(
                old, index=False
            )
            pd.DataFrame([{"protocol_version": "validation_first_v6"}]).to_csv(
                current, index=False
            )
            source = root / "sources.json"
            destination = root / "effective.json"
            source.write_text(
                json.dumps(
                    {
                        "protocol_version": "validation_first_v6",
                        "validation_inputs": [],
                        "final_inputs": [
                            {"label": "old", "path": str(old)},
                            {"label": "current", "path": str(current)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            provisional_source_config(source, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(
                [item["label"] for item in payload["final_inputs"]], ["current"]
            )


if __name__ == "__main__":
    unittest.main()
