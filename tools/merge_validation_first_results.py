"""Safely merge validation-first worker CSV files by their stable run ID.

Run this only after every process writing to the input or output files has
stopped. Exact duplicate rows are accepted once. If the same run ID carries
different values, the script stops instead of silently choosing one result.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_PROTOCOL_VERSION = "validation_first_v6"
REQUIRED_COLUMNS = {
    "protocol_version",
    "run_id",
    "scenario",
    "dataset",
    "model",
    "status",
}


def read_worker_file(path: Path, protocol_version: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    protocols = set(frame["protocol_version"])
    if protocol_version not in protocols:
        raise ValueError(
            f"{path} does not contain required protocol version {protocol_version}"
        )
    if (frame["run_id"].str.strip() == "").any():
        raise ValueError(f"{path} contains an empty run_id")
    return frame


def merge_worker_files(paths: list[Path], protocol_version: str) -> pd.DataFrame:
    if len(paths) < 2:
        raise ValueError("At least two input files are required")

    frames = [read_worker_file(path, protocol_version) for path in paths]
    columns: list[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)

    aligned = [frame.reindex(columns=columns, fill_value="") for frame in frames]
    rows_by_id: dict[str, tuple[str, ...]] = {}
    retained_rows: list[dict[str, str]] = []
    sources_by_id: dict[str, Path] = {}

    for path, frame in zip(paths, aligned, strict=True):
        for row in frame.to_dict(orient="records"):
            identifier = row["run_id"]
            canonical = tuple(row[column] for column in columns)
            previous = rows_by_id.get(identifier)
            if previous is None:
                rows_by_id[identifier] = canonical
                sources_by_id[identifier] = path
                retained_rows.append(row)
                continue
            if previous != canonical:
                different = [
                    column
                    for column, old, new in zip(
                        columns, previous, canonical, strict=True
                    )
                    if old != new
                ]
                raise ValueError(
                    "Conflicting duplicate run_id "
                    f"{identifier!r} in {sources_by_id[identifier]} and {path}; "
                    f"different columns: {', '.join(different)}"
                )

    return pd.DataFrame(retained_rows, columns=columns)


def write_atomically(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="worker CSV files")
    parser.add_argument("--output", required=True, type=Path, help="merged CSV file")
    parser.add_argument(
        "--protocol-version",
        default=DEFAULT_PROTOCOL_VERSION,
        help="protocol version that must occur in every input file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    output = args.output.resolve()
    merged = merge_worker_files(inputs, args.protocol_version)
    write_atomically(merged, output)

    selected = merged[merged["protocol_version"].eq(args.protocol_version)]
    successful = int(selected["status"].eq("success").sum())
    failed = int(selected["status"].eq("failed").sum())
    print(
        f"Merged {len(merged)} unique rows; {args.protocol_version} contains "
        f"{successful} successful and {failed} failed rows."
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
