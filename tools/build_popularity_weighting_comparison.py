from pathlib import Path

import pandas as pd


SOURCE_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "recbole_results"
    / "tuning_results"
    / "analysis_results"
    / "structured_report"
    / "session_full_tuning_results.csv"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "recbole_results"
    / "presentation_summary_2026_06_17"
)


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results = results[(results["status"] == "success") & (results["model"] != "model")]

    for column in [
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "runtime_seconds",
        "vsknn_popularity_weight",
        "vstan_popularity_weight",
    ]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    return results


def best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values("mrr@10", ascending=False).iloc[0]


def build_weighting_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    model_weight_columns = {
        "VS-KNN": "vsknn_popularity_weight",
        "VSTAN": "vstan_popularity_weight",
    }

    for model, weight_column in model_weight_columns.items():
        model_results = results[results["model"] == model].copy()

        for dataset, group in model_results.groupby("dataset"):
            explicit = group[group[weight_column].notna()].copy()
            unweighted = explicit[explicit[weight_column] == 0.0]
            weighted = explicit[explicit[weight_column] > 0.0]

            if unweighted.empty:
                unweighted = group[group[weight_column].isna()]

            if unweighted.empty or weighted.empty:
                continue

            best_unweighted = best_row(unweighted)
            best_weighted = best_row(weighted)

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "best_unweighted_mrr@10": best_unweighted["mrr@10"],
                    "best_weighted_mrr@10": best_weighted["mrr@10"],
                    "delta_mrr@10_weighted_minus_unweighted": (
                        best_weighted["mrr@10"] - best_unweighted["mrr@10"]
                    ),
                    "weighted_improves_mrr@10": (
                        best_weighted["mrr@10"] > best_unweighted["mrr@10"]
                    ),
                    "best_unweighted_hit@10": best_unweighted["hit@10"],
                    "best_weighted_hit@10": best_weighted["hit@10"],
                    "delta_hit@10_weighted_minus_unweighted": (
                        best_weighted["hit@10"] - best_unweighted["hit@10"]
                    ),
                    "best_unweighted_ndcg@10": best_unweighted["ndcg@10"],
                    "best_weighted_ndcg@10": best_weighted["ndcg@10"],
                    "delta_ndcg@10_weighted_minus_unweighted": (
                        best_weighted["ndcg@10"] - best_unweighted["ndcg@10"]
                    ),
                    "best_weighted_popularity_weight": best_weighted[
                        weight_column
                    ],
                    "best_unweighted_runtime_seconds": best_unweighted[
                        "runtime_seconds"
                    ],
                    "best_weighted_runtime_seconds": best_weighted[
                        "runtime_seconds"
                    ],
                    "best_unweighted_config": best_unweighted["config_json"],
                    "best_weighted_config": best_weighted["config_json"],
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "model"])


def build_weight_level_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    model_weight_columns = {
        "VS-KNN": "vsknn_popularity_weight",
        "VSTAN": "vstan_popularity_weight",
    }

    for model, weight_column in model_weight_columns.items():
        model_results = results[
            (results["model"] == model) & results[weight_column].notna()
        ].copy()

        for keys, group in model_results.groupby(["dataset", weight_column]):
            dataset, weight = keys
            best = best_row(group)

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "popularity_weight": weight,
                    "best_mrr@10": best["mrr@10"],
                    "best_hit@10": best["hit@10"],
                    "best_ndcg@10": best["ndcg@10"],
                    "runtime_seconds": best["runtime_seconds"],
                    "config_json": best["config_json"],
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["dataset", "model", "popularity_weight"]
    )


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No data available."

    text_frame = frame.fillna("").astype(str)
    headers = list(text_frame.columns)
    rows = text_frame.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Popularity Weighting Evaluation",
        "",
        "This table compares the best unweighted and weighted VS-KNN/VSTAN "
        "configuration per dataset.",
        "",
    ]

    if summary.empty:
        lines.append("No weighting comparison data available.")
    else:
        display = summary[
            [
                "dataset",
                "model",
                "best_unweighted_mrr@10",
                "best_weighted_mrr@10",
                "delta_mrr@10_weighted_minus_unweighted",
                "weighted_improves_mrr@10",
                "best_weighted_popularity_weight",
            ]
        ].copy()
        lines.append(to_markdown_table(display))
        lines.extend(
            [
                "",
                "Interpretation: negative delta values mean that popularity "
                "weighting reduced ranking quality compared with the unweighted "
                "variant.",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = clean_results(pd.read_csv(SOURCE_RESULTS))

    summary = build_weighting_summary(results)
    by_weight = build_weight_level_summary(results)

    summary_path = OUTPUT_DIR / "popularity_weighting_improvement_summary.csv"
    by_weight_path = OUTPUT_DIR / "popularity_weighting_by_weight_level.csv"
    markdown_path = OUTPUT_DIR / "popularity_weighting_evaluation.md"

    summary.to_csv(summary_path, index=False)
    by_weight.to_csv(by_weight_path, index=False)
    write_markdown(summary, markdown_path)

    print(summary_path)
    print(by_weight_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
