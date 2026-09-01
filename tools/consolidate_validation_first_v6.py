"""Consolidate validation-first v6 worker files and build paper-ready exports.

The source manifest identifies files by full path, so equal base names from
different worker directories are unambiguous. Validation duplicates are
resolved conservatively:

* a successful retry replaces one or more failed rows with the same ``run_id``;
* exact duplicates are retained once;
* successful validation duplicates may differ only in ``runtime_seconds``;
* different scientific values for the same successful ID are a hard error.

Final-test duplicates are stricter because runtime is part of the reported
result: any non-exact successful duplicate is a hard error.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.run_validation_first_experiments as experiments


DEFAULT_SOURCE_CONFIG = PROJECT_ROOT / "config" / "validation_first_v6_result_sources.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "recbole_results" / "validation_first_v6_consolidated"
VALIDATION_REQUIRED_COLUMNS = {
    "protocol_version",
    "evaluated_split",
    "run_id",
    "scenario",
    "dataset",
    "model",
    "status",
    "config_json",
}
FINAL_REQUIRED_COLUMNS = {
    "protocol_version",
    "evaluated_split",
    "final_test_id",
    "scenario",
    "dataset",
    "model",
    "seed",
    "status",
    "config_json",
}
VALIDATION_TOLERATED_SUCCESS_DIFFERENCES = {"runtime_seconds"}
DATASET_LABELS = {
    "movielens_recbole": "MovieLens",
    "amazon_recbole": "Amazon",
    "adressa_recbole_sample": "Adressa",
    "globo_recbole_sample": "Globo",
    "yoochoose_recbole_sample": "Yoochoose",
}


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path


@dataclass
class MergeResult:
    frame: pd.DataFrame
    decisions: pd.DataFrame
    conflicts: list[str]


def normalise(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomically(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_source_config(path: Path) -> tuple[str, list[SourceSpec], list[SourceSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = str(payload["protocol_version"])

    def specs(key: str) -> list[SourceSpec]:
        result = []
        for item in payload.get(key, []):
            source_path = Path(item["path"])
            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path
            result.append(SourceSpec(str(item["label"]), source_path.resolve()))
        return result

    return protocol, specs("validation_inputs"), specs("final_inputs")


def read_sources(
    specs: list[SourceSpec],
    protocol: str,
    required_columns: set[str],
    id_column: str,
    allow_incomplete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for order, spec in enumerate(specs):
        entry: dict[str, Any] = {
            "source_order": order,
            "source_label": spec.label,
            "source_path": str(spec.path),
            "file_name": spec.path.name,
            "exists": spec.path.is_file(),
        }
        if not spec.path.is_file():
            entry.update(
                {
                    "sha256": "",
                    "total_rows": 0,
                    "protocol_rows": 0,
                    "success_rows": 0,
                    "failed_rows": 0,
                    "state": "missing",
                }
            )
            inventory.append(entry)
            if not allow_incomplete:
                raise FileNotFoundError(f"Required result source is missing: {spec.path}")
            continue

        frame = pd.read_csv(
            spec.path, dtype=str, keep_default_na=False, low_memory=False
        )
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{spec.path} is missing required columns: {', '.join(missing_columns)}"
            )
        selected = frame[frame["protocol_version"].eq(protocol)].copy()
        if selected.empty and not allow_incomplete:
            raise ValueError(f"{spec.path} has no rows for protocol {protocol}")
        if not selected.empty and selected[id_column].str.strip().eq("").any():
            raise ValueError(f"{spec.path} contains an empty {id_column}")

        selected["_source_order"] = order
        selected["_source_label"] = spec.label
        selected["_source_path"] = str(spec.path)
        selected["_source_row"] = selected.index + 2
        rows.append(selected)
        entry.update(
            {
                "sha256": sha256(spec.path),
                "total_rows": len(frame),
                "protocol_rows": len(selected),
                "success_rows": int(selected["status"].eq("success").sum()),
                "failed_rows": int(selected["status"].eq("failed").sum()),
                "state": "loaded" if not selected.empty else "protocol_not_present",
            }
        )
        inventory.append(entry)

    combined = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return combined, pd.DataFrame(inventory)


def differing_columns(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if len({normalise(row.get(column, "")) for row in rows}) > 1
    ]


def consolidate_rows(
    frame: pd.DataFrame,
    id_column: str,
    tolerated_success_differences: set[str],
) -> MergeResult:
    if frame.empty:
        return MergeResult(pd.DataFrame(), pd.DataFrame(), [])

    internal_columns = [column for column in frame.columns if column.startswith("_")]
    data_columns = [column for column in frame.columns if column not in internal_columns]
    retained: list[dict[str, str]] = []
    decisions: list[dict[str, Any]] = []
    conflicts: list[str] = []

    for identifier, group in frame.groupby(id_column, sort=False, dropna=False):
        ordered = group.sort_values(
            ["_source_order", "_source_row"], kind="stable"
        )
        records = ordered.to_dict(orient="records")
        success_records = [row for row in records if row.get("status") == "success"]
        failed_records = [row for row in records if row.get("status") != "success"]
        decision = "unique"
        differences = differing_columns(records, data_columns)

        if success_records:
            selected = success_records[0]
            success_differences = differing_columns(success_records, data_columns)
            unacceptable = sorted(
                set(success_differences)
                - tolerated_success_differences
                - {"error"}
            )
            if unacceptable:
                message = (
                    f"Conflicting successful {id_column} {identifier!r}; "
                    f"different columns: {', '.join(unacceptable)}"
                )
                conflicts.append(message)
                continue
            if failed_records:
                decision = "success_replaces_failed"
            elif len(success_records) > 1 and success_differences:
                decision = "equivalent_success_prefer_first_source"
            elif len(success_records) > 1:
                decision = "exact_duplicate"
        else:
            selected = records[0]
            decision = "retained_failure" if len(records) == 1 else "multiple_failures_prefer_first_source"

        retained.append({column: normalise(selected.get(column, "")) for column in data_columns})
        decisions.append(
            {
                id_column: normalise(identifier),
                "decision": decision,
                "selected_status": normalise(selected.get("status")),
                "selected_source_label": normalise(selected.get("_source_label")),
                "selected_source_path": normalise(selected.get("_source_path")),
                "source_count": len(records),
                "success_source_count": len(success_records),
                "failed_source_count": len(failed_records),
                "all_source_labels": ";".join(
                    normalise(row.get("_source_label")) for row in records
                ),
                "differing_columns": ";".join(differences),
            }
        )

    if conflicts:
        raise ValueError("\n".join(conflicts))
    return MergeResult(
        pd.DataFrame(retained, columns=data_columns),
        pd.DataFrame(decisions),
        conflicts,
    )


def annotate_validation_selection(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark resource-excluded BPR configurations without deleting audit rows."""
    annotated = frame.copy()
    if annotated.empty:
        annotated["selection_eligible"] = pd.Series(dtype=bool)
        annotated["selection_exclusion_reason"] = pd.Series(dtype=str)
        return annotated

    def exclusion_reason(row: pd.Series) -> str:
        if row.get("scenario") != "topn" or row.get("model") != "BPR":
            return ""
        try:
            config = json.loads(normalise(row.get("config_json")))
            embedding_size = int(config.get("embedding_size", 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
        if embedding_size > experiments.MAX_BPR_EMBEDDING_SIZE:
            return (
                f"BPR embedding_size={embedding_size} exceeds the resource-bounded "
                f"maximum of {experiments.MAX_BPR_EMBEDDING_SIZE}"
            )
        return ""

    annotated["selection_exclusion_reason"] = annotated.apply(
        exclusion_reason, axis=1
    )
    annotated["selection_eligible"] = annotated[
        "selection_exclusion_reason"
    ].eq("")
    return annotated


def expected_validation_rows() -> pd.DataFrame:
    rows = []
    for scenario, dataset, model, config in experiments.expected_trials(
        ["topn", "session"], None, None
    ):
        rows.append(
            {
                "run_id": experiments.run_id(scenario, dataset, model, config),
                "scenario": scenario,
                "dataset": dataset,
                "model": model,
                "config_json": experiments.serialise_config(config),
            }
        )
    return pd.DataFrame(rows)


def expected_final_rows() -> pd.DataFrame:
    rows = []
    for scenario, datasets, models in [
        ("topn", experiments.TOPN_DATASETS, experiments.TOPN_MODELS),
        ("session", experiments.SESSION_DATASETS, experiments.SESSION_MODELS),
    ]:
        for dataset in datasets:
            for model in models:
                for seed in experiments.final_seeds(model):
                    rows.append(
                        {
                            "final_test_id": experiments.final_test_id(
                                scenario, dataset, model, seed
                            ),
                            "scenario": scenario,
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def expected_optional_final_rows() -> pd.DataFrame:
    rows = []
    for scenario, datasets, models in [
        ("topn", experiments.TOPN_DATASETS, experiments.TOPN_MODELS),
        ("session", experiments.SESSION_DATASETS, experiments.SESSION_MODELS),
    ]:
        for dataset in datasets:
            for model in models:
                for seed in experiments.optional_final_seeds(model):
                    rows.append(
                        {
                            "final_test_id": experiments.final_test_id(
                                scenario, dataset, model, seed
                            ),
                            "scenario": scenario,
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def completion_table(
    expected: pd.DataFrame,
    merged: pd.DataFrame,
    id_column: str,
    stage: str,
    allowed_additional_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    successful_ids = (
        set(merged.loc[merged["status"].eq("success"), id_column])
        if not merged.empty
        else set()
    )
    expected_ids = set(expected[id_column])
    allowed_ids = allowed_additional_ids or set()
    missing_ids = expected_ids - successful_ids
    unexpected_ids = sorted(
        set(merged[id_column]) - expected_ids - allowed_ids
        if not merged.empty
        else set()
    )
    expected_merged = (
        merged[merged[id_column].isin(expected_ids)].copy()
        if not merged.empty
        else merged
    )
    missing = expected[expected[id_column].isin(missing_ids)].copy()
    missing.insert(0, "stage", stage)

    base = expected[["scenario", "dataset", "model"]].drop_duplicates()
    if expected_merged.empty:
        success_counts = pd.DataFrame(columns=["scenario", "dataset", "model", "successful"])
        failed_counts = pd.DataFrame(columns=["scenario", "dataset", "model", "failed"])
    else:
        success_counts = (
            expected_merged[expected_merged["status"].eq("success")]
            .groupby(["scenario", "dataset", "model"], as_index=False)
            .size()
            .rename(columns={"size": "successful"})
        )
        failed_counts = (
            expected_merged[expected_merged["status"].ne("success")]
            .groupby(["scenario", "dataset", "model"], as_index=False)
            .size()
            .rename(columns={"size": "failed"})
        )
    expected_counts = (
        expected.groupby(["scenario", "dataset", "model"], as_index=False)
        .size()
        .rename(columns={"size": "expected"})
    )
    status = base.merge(expected_counts, on=["scenario", "dataset", "model"], how="left")
    status = status.merge(success_counts, on=["scenario", "dataset", "model"], how="left")
    status = status.merge(failed_counts, on=["scenario", "dataset", "model"], how="left")
    for column in ["successful", "failed"]:
        status[column] = (
            pd.to_numeric(status[column], errors="coerce").fillna(0).astype(int)
        )
    status["missing"] = status["expected"] - status["successful"]
    status.insert(0, "stage", stage)
    return status, missing, unexpected_ids


def select_validation_winners(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    annotated = annotate_validation_selection(frame)
    successful = annotated[
        annotated["status"].eq("success") & annotated["selection_eligible"]
    ].copy()
    successful["valid_mrr@10"] = pd.to_numeric(
        successful["valid_mrr@10"], errors="coerce"
    )
    successful["runtime_seconds"] = pd.to_numeric(
        successful["runtime_seconds"], errors="coerce"
    ).fillna(float("inf"))
    successful = successful.dropna(subset=["valid_mrr@10"])
    ranked = successful.sort_values(
        ["scenario", "dataset", "model", "valid_mrr@10", "runtime_seconds", "run_id"],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    )
    winners = ranked.groupby(
        ["scenario", "dataset", "model"], as_index=False, sort=True
    ).head(1)
    return winners.reset_index(drop=True)


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: None
                if value is None or (not isinstance(value, (list, dict)) and pd.isna(value))
                else value.item()
                if hasattr(value, "item")
                else value
                for key, value in row.items()
            }
        )
    return records


def latex_escape(value: Any) -> str:
    text = normalise(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_metric(row: pd.Series, metric: str) -> str:
    mean = pd.to_numeric(pd.Series([row.get(f"{metric}_mean")]), errors="coerce").iloc[0]
    std = pd.to_numeric(pd.Series([row.get(f"{metric}_std")]), errors="coerce").iloc[0]
    if pd.isna(mean):
        return "--"
    if int(row.get("expected_seed_count", 1)) > 1 and not pd.isna(std):
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"{mean:.4f}"


def write_latex_quality_table(frame: pd.DataFrame, scenario: str, path: Path) -> None:
    subset = frame[frame["scenario"].eq(scenario)].copy() if not frame.empty else frame
    lines = [
        "% Generated by tools/consolidate_validation_first_v6.py; do not edit manually.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Model & MRR@10 & Hit@10 & NDCG@10 & Coverage@10 \\",
        r"\midrule",
    ]
    for _, row in subset.sort_values(["dataset", "model"]).iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(DATASET_LABELS.get(str(row["dataset"]), row["dataset"])),
                    latex_escape(row["model"]),
                    format_metric(row, "mrr@10"),
                    format_metric(row, "hit@10"),
                    format_metric(row, "ndcg@10"),
                    format_metric(row, "coverage@10"),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Final {latex_escape(scenario)} recommendation results.}}",
            rf"\label{{tab:final-{latex_escape(scenario)}-results}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_workbook_payload(
    audit: dict[str, Any],
    inventory: pd.DataFrame,
    validation_status: pd.DataFrame,
    validation_merged: pd.DataFrame,
    validation_excluded: pd.DataFrame,
    selected: pd.DataFrame,
    final_merged: pd.DataFrame,
    final_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    missing: pd.DataFrame,
) -> dict[str, Any]:
    summary_rows = [
        {"item": "Protocol", "value": audit["protocol_version"]},
        {"item": "Generated at", "value": audit["generated_at"]},
        {"item": "Validation expected", "value": audit["validation"]["expected"]},
        {"item": "Validation successful", "value": audit["validation"]["successful"]},
        {"item": "Validation complete", "value": audit["validation"]["complete"]},
        {"item": "Final tests expected", "value": audit["final_tests"]["expected"]},
        {"item": "Final tests successful", "value": audit["final_tests"]["successful"]},
        {"item": "Final tests complete", "value": audit["final_tests"]["complete"]},
        {"item": "Overall state", "value": audit["state"]},
    ]
    sheets = {
        "README": summary_rows,
        "Source Inventory": clean_records(inventory),
        "Validation Status": clean_records(validation_status),
        "Selected Configs": clean_records(selected),
        "Validation Merged": clean_records(validation_merged),
        "Validation Excluded": clean_records(validation_excluded),
        "Final Raw": clean_records(final_merged),
        "Final Summary": clean_records(final_summary),
        "Merge Decisions": clean_records(decisions),
        "Missing Runs": clean_records(missing),
    }
    return {"title": "Validation-first v6 Results", "audit": audit, "sheets": sheets}


def run_pipeline(
    source_config: Path,
    output_dir: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    protocol, validation_specs, final_specs = load_source_config(source_config)
    if protocol != experiments.PROTOCOL_VERSION:
        raise ValueError(
            f"Source config requests {protocol}, runner uses {experiments.PROTOCOL_VERSION}"
        )

    validation_raw, validation_inventory = read_sources(
        validation_specs,
        protocol,
        VALIDATION_REQUIRED_COLUMNS,
        "run_id",
        allow_incomplete,
    )
    validation_merge = consolidate_rows(
        validation_raw,
        "run_id",
        VALIDATION_TOLERATED_SUCCESS_DIFFERENCES,
    )
    validation_merge.frame = annotate_validation_selection(validation_merge.frame)
    validation_eligible = validation_merge.frame[
        validation_merge.frame["selection_eligible"]
    ].copy()
    validation_excluded = validation_merge.frame[
        ~validation_merge.frame["selection_eligible"]
    ].copy()
    expected_validation = expected_validation_rows()
    validation_status, validation_missing, validation_unexpected = completion_table(
        expected_validation, validation_eligible, "run_id", "validation"
    )
    selected = select_validation_winners(validation_merge.frame)

    final_raw, final_inventory = read_sources(
        final_specs,
        protocol,
        FINAL_REQUIRED_COLUMNS,
        "final_test_id",
        allow_incomplete,
    )
    final_merge = consolidate_rows(final_raw, "final_test_id", set())
    expected_final = expected_final_rows()
    optional_final = expected_optional_final_rows()
    optional_final_ids = set(optional_final["final_test_id"])
    final_status, final_missing, final_unexpected = completion_table(
        expected_final,
        final_merge.frame,
        "final_test_id",
        "final_test",
        allowed_additional_ids=optional_final_ids,
    )
    final_successes = (
        final_merge.frame[final_merge.frame["status"].eq("success")].copy()
        if not final_merge.frame.empty
        else pd.DataFrame()
    )
    final_summary = (
        experiments.build_final_test_summary(final_successes)
        if not final_successes.empty
        else pd.DataFrame()
    )

    inventory = pd.concat(
        [
            validation_inventory.assign(stage="validation"),
            final_inventory.assign(stage="final_test"),
        ],
        ignore_index=True,
        sort=False,
    )
    decisions = pd.concat(
        [
            validation_merge.decisions.assign(stage="validation"),
            final_merge.decisions.assign(stage="final_test"),
        ],
        ignore_index=True,
        sort=False,
    )
    missing = pd.concat([validation_missing, final_missing], ignore_index=True, sort=False)
    status = pd.concat([validation_status, final_status], ignore_index=True, sort=False)

    validation_success_count = int(
        validation_eligible["status"].eq("success").sum()
        if not validation_eligible.empty
        else 0
    )
    expected_final_ids = set(expected_final["final_test_id"])
    successful_final_ids = (
        set(
            final_merge.frame.loc[
                final_merge.frame["status"].eq("success"), "final_test_id"
            ]
        )
        if not final_merge.frame.empty
        else set()
    )
    final_success_count = len(successful_final_ids.intersection(expected_final_ids))
    optional_final_success_count = len(
        successful_final_ids.intersection(optional_final_ids)
    )
    all_final_success_count = len(successful_final_ids)
    validation_complete = (
        validation_success_count == len(expected_validation) and not validation_unexpected
    )
    final_complete = final_success_count == len(expected_final) and not final_unexpected
    state = "complete" if validation_complete and final_complete else "incomplete"
    audit = {
        "protocol_version": protocol,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "state": state,
        "allow_incomplete": allow_incomplete,
        "source_config": str(source_config.resolve()),
        "selection_policy": {
            "max_bpr_embedding_size": experiments.MAX_BPR_EMBEDDING_SIZE,
            "primary_final_seeds": list(experiments.PRIMARY_FINAL_EVALUATION_SEEDS),
            "optional_robustness_seeds": list(experiments.OPTIONAL_ROBUSTNESS_SEEDS),
        },
        "validation": {
            "expected": len(expected_validation),
            "successful": validation_success_count,
            "all_merged_successful": int(
                validation_merge.frame["status"].eq("success").sum()
                if not validation_merge.frame.empty
                else 0
            ),
            "excluded_by_budget_rows": len(validation_excluded),
            "excluded_by_budget_successful": int(
                validation_excluded["status"].eq("success").sum()
                if not validation_excluded.empty
                else 0
            ),
            "failed_retained": int(
                validation_eligible["status"].ne("success").sum()
                if not validation_eligible.empty
                else 0
            ),
            "missing_successes": len(validation_missing),
            "unexpected_ids": validation_unexpected,
            "selected_config_count": len(selected),
            "expected_selected_config_count": 26,
            "complete": validation_complete,
        },
        "final_tests": {
            "expected": len(expected_final),
            "successful": final_success_count,
            "all_successful": all_final_success_count,
            "optional_expected": len(optional_final),
            "optional_successful": optional_final_success_count,
            "failed_retained": int(
                (
                    final_merge.frame["final_test_id"].isin(expected_final_ids)
                    & final_merge.frame["status"].ne("success")
                ).sum()
                if not final_merge.frame.empty
                else 0
            ),
            "missing_successes": len(final_missing),
            "unexpected_ids": final_unexpected,
            "summary_group_count": len(final_summary),
            "expected_summary_group_count": 26,
            "complete": final_complete,
        },
        "merge_decisions": dict(Counter(decisions.get("decision", pd.Series(dtype=str)))),
    }

    if not allow_incomplete and state != "complete":
        raise RuntimeError(
            "Result consolidation is incomplete: "
            f"validation {validation_success_count}/{len(expected_validation)}, "
            f"final tests {final_success_count}/{len(expected_final)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomically(validation_merge.frame, output_dir / "validation_trials_merged.csv")
    write_csv_atomically(
        validation_eligible[validation_eligible["status"].eq("success")],
        output_dir / "validation_successes.csv",
    )
    write_csv_atomically(
        validation_excluded, output_dir / "validation_excluded_by_budget.csv"
    )
    write_csv_atomically(selected, output_dir / "selected_validation_configs.csv")
    write_csv_atomically(final_merge.frame, output_dir / "final_test_results_merged.csv")
    write_csv_atomically(final_summary, output_dir / "final_test_summary.csv")
    write_csv_atomically(inventory, output_dir / "source_inventory.csv")
    write_csv_atomically(decisions, output_dir / "merge_decisions.csv")
    write_csv_atomically(status, output_dir / "completion_status.csv")
    write_csv_atomically(missing, output_dir / "missing_runs.csv")
    write_json_atomically(audit, output_dir / "audit_report.json")
    write_json_atomically(
        build_workbook_payload(
            audit,
            inventory,
            validation_status,
            validation_merge.frame,
            validation_excluded,
            selected,
            final_merge.frame,
            final_summary,
            decisions,
            missing,
        ),
        output_dir / "workbook_payload.json",
    )
    write_latex_quality_table(final_summary, "topn", output_dir / "final_topn_results.tex")
    write_latex_quality_table(final_summary, "session", output_dir / "final_session_results.tex")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a provisional audit even when sources or expected runs are missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = run_pipeline(
        args.source_config.resolve(), args.output_dir.resolve(), args.allow_incomplete
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
