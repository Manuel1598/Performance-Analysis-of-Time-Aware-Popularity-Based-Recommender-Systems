"""Build the consolidated best-model overview used by the thesis documentation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]
OPTIONAL_COLUMNS = ["coverage@10", "avg_recommendation_popularity@10"]


def clean_successful_results(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned[
        cleaned["status"].eq("success")
        & cleaned["model"].notna()
        & cleaned["dataset"].notna()
        & cleaned["model"].ne("model")
        & cleaned["dataset"].ne("dataset")
    ]
    for column in [*METRIC_COLUMNS, *OPTIONAL_COLUMNS, "runtime_seconds"]:
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned[cleaned["mrr@10"].notna()].copy()


def select_best_per_model_dataset(data: pd.DataFrame) -> pd.DataFrame:
    ranked = data.copy()
    ranked["_source_order"] = range(len(ranked))
    ranked["_runtime_rank"] = ranked.get(
        "runtime_seconds", pd.Series(float("inf"), index=ranked.index)
    ).fillna(float("inf"))
    ranked = ranked.sort_values(
        ["dataset", "model", "mrr@10", "_runtime_rank", "_source_order"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    return (
        ranked.groupby(["dataset", "model"], as_index=False, sort=True)
        .head(1)
        .drop(columns=["_source_order", "_runtime_rank"])
        .reset_index(drop=True)
    )


def audited_vsknn_rows(best_file: Path) -> pd.DataFrame:
    data = pd.read_csv(best_file)
    data = data.rename(columns={"model": "legacy_model_label"})
    data["model"] = "VSKNN"
    data["scenario"] = "Session-based"
    data["implementation"] = "audited"
    data["result_source"] = "compact_tuning_best_by_dataset.csv"
    data["config_json"] = data.apply(
        lambda row: json.dumps(
            {
                "neighbor_size": int(row["neighbor_size"]),
                "sample_size": int(row["sample_size"]),
                "sampling": row["sampling"],
                "similarity": row["similarity"],
                "session_weighting": row["session_weighting"],
                "score_weighting": row["score_weighting"],
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return data


def build_overview(project_root: Path) -> pd.DataFrame:
    tuning_dir = project_root / "recbole_results" / "tuning_results"
    audited_dir = project_root / "recbole_results" / "vsknn_audited"

    topn = clean_successful_results(
        pd.read_csv(tuning_dir / "topn_full_tuning_results.csv")
    )
    topn["scenario"] = "Top-N"
    topn["implementation"] = "existing"
    topn["result_source"] = "topn_full_tuning_results.csv"

    session = clean_successful_results(
        pd.read_csv(tuning_dir / "session_full_tuning_results.csv")
    )
    session = session[session["model"].ne("VS-KNN")].copy()
    if "vstan_popularity_weight" in session:
        current_vstan_mask = (
            session["model"].ne("VSTAN")
            | session["vstan_popularity_weight"].notna()
        )
        session = session[current_vstan_mask].copy()
    session["scenario"] = "Session-based"
    session["implementation"] = "existing"
    session["result_source"] = "session_full_tuning_results.csv"

    selected = pd.concat(
        [
            select_best_per_model_dataset(topn),
            select_best_per_model_dataset(session),
            audited_vsknn_rows(
                audited_dir / "compact_tuning_best_by_dataset.csv"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    columns = [
        "scenario",
        "dataset",
        "model",
        "implementation",
        "device",
        *METRIC_COLUMNS,
        *OPTIONAL_COLUMNS,
        "runtime_seconds",
        "config_json",
        "result_source",
        "run_id",
    ]
    for column in columns:
        if column not in selected:
            selected[column] = pd.NA
    selected = selected[columns]
    return selected.sort_values(
        ["scenario", "dataset", "mrr@10", "model"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def best_models_by_dataset(overview: pd.DataFrame) -> pd.DataFrame:
    ranked = overview.sort_values(
        ["scenario", "dataset", "mrr@10", "model"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return ranked.groupby(["scenario", "dataset"], as_index=False).head(1)


def markdown_table(data: pd.DataFrame) -> str:
    display = data[["model", "hit@10", "ndcg@10", "mrr@10", "runtime_seconds"]].copy()
    for column in ["hit@10", "ndcg@10", "mrr@10"]:
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.4f}"
        )
    display["runtime_seconds"] = display["runtime_seconds"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.2f}"
    )
    display.columns = ["Model", "Hit@10", "NDCG@10", "MRR@10", "Runtime (s)"]
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def write_markdown(project_root: Path, overview: pd.DataFrame, output_file: Path) -> None:
    winners = best_models_by_dataset(overview)
    lines = [
        "# Audited VSKNN changes and consolidated best-model overview",
        "",
        "## Purpose and selection rule",
        "",
        "This file consolidates the best successful configuration of every model on every",
        "available dataset. Selection uses the thesis primary metric **MRR@10**. If MRR@10",
        "ties, the lower recorded runtime and then the original row order are used.",
        "",
        "Legacy VS-KNN rows are deliberately excluded and replaced by the corrected,",
        "performance-optimized, compact-tuned **audited VSKNN** rows.",
        "",
        "## What changed in VSKNN and why",
        "",
        "- Replaced unweighted cosine SKNN-like scoring with reference VSKNN position weighting.",
        "- Added neighbor score decay based on the most recent shared click.",
        "- Collapsed RecBole prefix-target augmentation to one training reference session per",
        "  session ID, preventing duplicated longer sessions from being over-weighted.",
        "- Kept the index restricted to `train_data.dataset`; validation/test targets are not",
        "  inserted into the neighbor index.",
        "- Removed the thesis-specific popularity correction from the upstream-faithful path.",
        "- Standardized class/configuration names and retained a temporary compatibility alias.",
        "- Replaced per-query global candidate sorting with a lazy merge of pre-sorted item",
        "  session lists; quality is unchanged while runtime is substantially lower.",
        "- Added algorithm, leakage, candidate-order, CLI, resume, and best-selection tests.",
        "- Retuned only the corrected VSKNN; unchanged models and data splits were not rerun.",
        "",
        "## Important comparability notes",
        "",
        "- Ranking metrics can be compared within the recorded evaluation setup.",
        "- Runtime is descriptive only when device/hardware differs. Audited VSKNN sample runs",
        "  use CPU, while many stored legacy/framework results use CUDA.",
        "- Top-N datasets and session datasets represent different recommendation scenarios and",
        "  should not be ranked against each other.",
        "- Session results currently use the three local sample datasets; Top-N results use the",
        "  stored MovieLens and Amazon RecBole datasets.",
        "",
        "## Best configuration of every model by dataset",
        "",
    ]

    for (scenario, dataset), group in overview.groupby(["scenario", "dataset"], sort=True):
        winner = winners[
            winners["scenario"].eq(scenario) & winners["dataset"].eq(dataset)
        ].iloc[0]
        lines.extend(
            [
                f"### {scenario}: `{dataset}`",
                "",
                markdown_table(group),
                "",
                f"**Best by MRR@10:** {winner['model']} ({winner['mrr@10']:.4f}).",
                "",
            ]
        )

    lines.extend(
        [
            "## Source files",
            "",
            "- `recbole_results/tuning_results/topn_full_tuning_results.csv`",
            "- `recbole_results/tuning_results/session_full_tuning_results.csv`",
            "- `recbole_results/vsknn_audited/compact_tuning_best_by_dataset.csv`",
            "- generated detailed CSV: `recbole_results/summary/best_models_by_dataset.csv`",
            "- generated dataset winners: `recbole_results/summary/best_overall_model_per_dataset.csv`",
            "",
            "Regenerate this overview with:",
            "",
            "```powershell",
            "python -m src.recbole_framework.analysis.build_best_model_overview",
            "```",
            "",
        ]
    )
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    summary_dir = project_root / "recbole_results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    overview = build_overview(project_root)
    winners = best_models_by_dataset(overview)
    overview.to_csv(summary_dir / "best_models_by_dataset.csv", index=False)
    winners.to_csv(summary_dir / "best_overall_model_per_dataset.csv", index=False)
    write_markdown(
        project_root,
        overview,
        project_root / "docs" / "audited_vsknn_and_best_models_overview.md",
    )
    print(f"Model-dataset rows: {len(overview)}")
    print(f"Dataset winners: {len(winners)}")
    print(winners[["scenario", "dataset", "model", "mrr@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
