from pathlib import Path

import pandas as pd


SOURCE_WORKBOOK = Path(
    r"C:\Users\manue\Downloads\recbole_evaluation_tables.xlsx"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "recbole_results"
    / "presentation_summary_2026_06_17"
)


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def available(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


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


def clean_experiment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "experiment_type" not in frame.columns:
        return frame.copy()

    cleaned = frame.copy()
    cleaned = cleaned[
        cleaned["experiment_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["session", "topn"])
    ]

    if "model" in cleaned.columns:
        cleaned = cleaned[cleaned["model"].notna()]

    if "dataset" in cleaned.columns:
        cleaned = cleaned[cleaned["dataset"].notna()]

    return cleaned.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(SOURCE_WORKBOOK, sheet_name=None)

    best = sheets["Best Configuration per Model"].copy()
    comparison = sheets["Quality and Runtime Comparison"].copy()
    runtime = sheets["Runtime Summary"].copy()
    tuning = sheets["Tuning Search Space"].copy()
    datasets = sheets["Dataset Characteristics"].copy()

    for frame in [best, comparison, runtime, tuning, datasets]:
        frame.columns = [str(column).strip() for column in frame.columns]

    best = clean_experiment_frame(best)
    comparison = clean_experiment_frame(comparison)
    runtime = clean_experiment_frame(runtime)
    tuning = clean_experiment_frame(tuning)

    metric_columns = [
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "runtime_seconds",
        "runtime_minutes",
        "mrr@10_per_minute",
        "rank_by_mrr@10",
        "relative_mrr@10_to_dataset_best",
        "mrr@10_gap_to_dataset_best",
    ]
    numeric(best, metric_columns)
    numeric(comparison, metric_columns)
    numeric(
        runtime,
        [
            "runtime_seconds_mean",
            "runtime_seconds_median",
            "runtime_seconds_min",
            "runtime_seconds_max",
        ],
    )

    key_columns = [
        "experiment_type",
        "dataset",
        "model",
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "coverage@10",
        "avg_recommendation_popularity@10",
        "runtime_seconds",
        "runtime_minutes",
        "mrr@10_per_minute",
        "rank_by_mrr@10",
        "relative_mrr@10_to_dataset_best",
        "mrr@10_gap_to_dataset_best",
        "quality_runtime_pareto_efficient",
    ]
    config_columns = [
        "vsknn_k",
        "vsknn_sample_size",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "vstan_idf_weighting",
        "hidden_size",
        "learning_rate",
        "dropout_prob",
        "epochs",
        "window_days",
        "decay_lambda",
        "embedding_size",
    ]

    compact_columns = available(best, key_columns + config_columns)

    best_concise = best[compact_columns].sort_values(
        ["experiment_type", "dataset", "rank_by_mrr@10", "model"],
        na_position="last",
    )
    best_concise.to_csv(
        OUTPUT_DIR / "best_config_per_model_dataset.csv",
        index=False,
    )

    best_dataset = best.sort_values("mrr@10", ascending=False).drop_duplicates(
        ["experiment_type", "dataset"],
        keep="first",
    )
    best_dataset[available(best_dataset, key_columns + config_columns)].sort_values(
        ["experiment_type", "dataset"]
    ).to_csv(OUTPUT_DIR / "best_model_per_dataset.csv", index=False)

    rankings = comparison[available(comparison, key_columns)].sort_values(
        ["experiment_type", "dataset", "rank_by_mrr@10", "model"],
        na_position="last",
    )
    rankings.to_csv(OUTPUT_DIR / "model_ranking_per_dataset.csv", index=False)

    session_rankings = rankings[rankings["experiment_type"].eq("session")]
    session_rankings.to_csv(
        OUTPUT_DIR / "session_model_ranking_per_dataset.csv",
        index=False,
    )

    topn_rankings = rankings[rankings["experiment_type"].eq("topn")]
    topn_rankings.to_csv(
        OUTPUT_DIR / "topn_model_ranking_per_dataset.csv",
        index=False,
    )

    popularity = best[best["model"].isin(["MostPop", "RecentPop", "DecayPop"])]
    popularity[available(popularity, key_columns + config_columns)].sort_values(
        ["experiment_type", "dataset", "model"]
    ).to_csv(OUTPUT_DIR / "popularity_baselines_best_configs.csv", index=False)

    session_knn = best[best["model"].isin(["VS-KNN", "VSTAN"])]
    session_knn[available(session_knn, key_columns + config_columns)].sort_values(
        ["dataset", "model"]
    ).to_csv(OUTPUT_DIR / "session_knn_best_configs.csv", index=False)

    runtime_columns = available(
        runtime,
        [
            "experiment_type",
            "dataset",
            "model",
            "runtime_seconds_count",
            "runtime_seconds_mean",
            "runtime_seconds_median",
            "runtime_seconds_min",
            "runtime_seconds_max",
            "eval_runtime_seconds_mean",
            "train_runtime_seconds_mean",
        ],
    )
    runtime[runtime_columns].sort_values(
        ["experiment_type", "dataset", "runtime_seconds_median", "model"],
        na_position="last",
    ).to_csv(OUTPUT_DIR / "runtime_summary_compact.csv", index=False)

    fastest = (
        runtime.sort_values("runtime_seconds_median")
        .groupby(["experiment_type", "dataset"], as_index=False)
        .first()
    )
    fastest[runtime_columns].sort_values(["experiment_type", "dataset"]).to_csv(
        OUTPUT_DIR / "fastest_model_per_dataset.csv",
        index=False,
    )

    datasets.to_csv(OUTPUT_DIR / "dataset_characteristics.csv", index=False)
    tuning.to_csv(OUTPUT_DIR / "tuning_search_space.csv", index=False)

    summary_lines = [
        "# Presentation Summary - RecBole Results",
        "",
        "## Best Model Per Dataset",
        best_dataset[
            available(
                best_dataset,
                [
                    "experiment_type",
                    "dataset",
                    "model",
                    "mrr@10",
                    "hit@10",
                    "ndcg@10",
                    "runtime_seconds",
                ],
            )
        ]
        .sort_values(["experiment_type", "dataset"])
        .pipe(to_markdown_table),
        "",
        "## Session Ranking",
        session_rankings[
            available(
                session_rankings,
                [
                    "dataset",
                    "model",
                    "mrr@10",
                    "hit@10",
                    "ndcg@10",
                    "runtime_seconds",
                    "rank_by_mrr@10",
                ],
            )
        ].pipe(to_markdown_table),
        "",
        "## Interpretation Notes",
        "- `MRR@10` is the primary ranking metric.",
        "- Session popularity baselines are included as MostPop, RecentPop, and DecayPop.",
        "- VS-KNN and VSTAN are compared against GRU4Rec under the same session protocol.",
        "- Use runtime tables to discuss quality-runtime tradeoffs.",
    ]
    (OUTPUT_DIR / "presentation_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
