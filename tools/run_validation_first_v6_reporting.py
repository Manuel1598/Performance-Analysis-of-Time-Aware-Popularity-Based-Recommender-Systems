"""Run the complete validation-first v6 reporting pipeline.

This is the stable user-facing entry point. In provisional mode it ignores an
existing final-result CSV that contains only an older protocol version. Once v6
rows exist, the source is passed to the strict consolidator unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import pandas as pd

from tools import consolidate_validation_first_v6 as core


def source_contains_protocol(path: Path, protocol: str) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "protocol_version" not in reader.fieldnames:
            return False
        return any(row.get("protocol_version") == protocol for row in reader)


def provisional_source_config(source_config: Path, destination: Path) -> Path:
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    protocol = str(payload["protocol_version"])
    retained = []
    for source in payload.get("final_inputs", []):
        path = Path(source["path"])
        if not path.is_absolute():
            path = core.PROJECT_ROOT / path
        if source_contains_protocol(path, protocol):
            retained.append(source)
    payload["final_inputs"] = retained
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def write_latex_allowing_empty(
    frame: pd.DataFrame, scenario: str, path: Path
) -> None:
    if frame.empty:
        frame = pd.DataFrame(columns=["scenario", "dataset", "model"])
    core._original_write_latex_quality_table(frame, scenario, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config", type=Path, default=core.DEFAULT_SOURCE_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=core.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config = args.source_config.resolve()
    output_dir = args.output_dir.resolve()

    core._original_write_latex_quality_table = core.write_latex_quality_table
    core.write_latex_quality_table = write_latex_allowing_empty

    with tempfile.TemporaryDirectory(prefix="validation_first_v6_reporting_") as tmp:
        effective_config = source_config
        if args.allow_incomplete:
            effective_config = provisional_source_config(
                source_config, Path(tmp) / "sources.json"
            )
        audit = core.run_pipeline(
            effective_config, output_dir, args.allow_incomplete
        )
    audit["requested_source_config"] = str(source_config)
    core.write_json_atomically(audit, output_dir / "audit_report.json")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
