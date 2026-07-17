from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SESSION_DATASET_MARKERS = ("adressa", "globo", "yoochoose")
POPULARITY_MODELS = ("MostPop", "RecentPop", "DecayPop")
SESSION_NEIGHBORHOOD_MODELS = ("VS-KNN", "VSTAN")


class ModelBehaviorExplainer:
    """Create diagnostics that help explain dataset-specific model behavior."""

    def __init__(
        self,
        project_root: Path | None = None,
        structured_report_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.structured_report_dir = (
            structured_report_dir
            or self.project_root
            / "recbole_results"
            / "tuning_results"
            / "analysis_results"
            / "structured_report"
        )
        self.output_dir = (
            output_dir
            or self.project_root
            / "recbole_results"
            / "tuning_results"
            / "analysis_results"
            / "model_behavior_analysis"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Path]:
        best_per_model = self._read_csv("best_per_model.csv")
        comparative_summary = self._read_csv("comparative_summary.csv")
        popularity_weighting = self._read_csv("popularity_weighting_summary.csv")

        datasets = sorted(
            dataset
            for dataset in comparative_summary["dataset"].dropna().unique()
            if isinstance(dataset, str)
        )

        dataset_behavior = self.dataset_behavior_summary(datasets)
        popular_items = self.popular_items(datasets)
        model_explanation = self.model_explanation_summary(
            comparative_summary=comparative_summary,
            dataset_behavior=dataset_behavior,
        )
        knn_vstan_explanation = self.knn_vstan_explanation(
            best_per_model=best_per_model,
            popularity_weighting=popularity_weighting,
            dataset_behavior=dataset_behavior,
        )
        pop_baseline_explanation = self.popularity_baseline_explanation(
            comparative_summary=comparative_summary,
            dataset_behavior=dataset_behavior,
        )

        paths = self.write_outputs(
            dataset_behavior=dataset_behavior,
            popular_items=popular_items,
            model_explanation=model_explanation,
            knn_vstan_explanation=knn_vstan_explanation,
            pop_baseline_explanation=pop_baseline_explanation,
        )
        plot_paths = self.create_plots(dataset_behavior, popular_items)
        paths.update(plot_paths)
        paths["report"] = self.write_report(
            dataset_behavior=dataset_behavior,
            model_explanation=model_explanation,
            knn_vstan_explanation=knn_vstan_explanation,
            pop_baseline_explanation=pop_baseline_explanation,
            plot_paths=plot_paths,
        )
        return paths

    def dataset_behavior_summary(self, datasets: list[str]) -> pd.DataFrame:
        rows = []
        for dataset in datasets:
            data = self._load_interactions(dataset)
            user_col = self._find_column(data, "user_id")
            item_col = self._find_column(data, "item_id")
            timestamp_col = self._find_column(data, "timestamp")

            item_counts = data[item_col].value_counts()
            user_counts = data[user_col].value_counts()
            interactions = len(data)
            items = int(item_counts.size)
            users = int(user_counts.size)

            top1_share = self._top_share(item_counts, 1)
            top10_share = self._top_share(item_counts, 10)
            top100_share = self._top_share(item_counts, 100)
            top1pct_share = self._top_fraction_share(item_counts, 0.01)
            gini = self._gini(item_counts)
            entropy = self._normalized_entropy(item_counts)
            repeat_item_share = self._repeat_item_share(data, user_col, item_col)

            row = {
                "dataset": dataset,
                "experiment_type": self._experiment_type(dataset),
                "interactions": interactions,
                "users_or_sessions": users,
                "items": items,
                "avg_interactions_per_user_or_session": interactions / users,
                "median_interactions_per_user_or_session": float(user_counts.median()),
                "p90_interactions_per_user_or_session": float(
                    user_counts.quantile(0.90)
                ),
                "avg_interactions_per_item": interactions / items,
                "interaction_density": interactions / (users * items),
                "top1_item_share": top1_share,
                "top10_item_share": top10_share,
                "top100_item_share": top100_share,
                "top1_percent_item_share": top1pct_share,
                "item_popularity_gini": gini,
                "item_popularity_normalized_entropy": entropy,
                "repeat_item_share_within_user_or_session": repeat_item_share,
                "popularity_profile": self._popularity_profile(
                    top10_share=top10_share,
                    top1pct_share=top1pct_share,
                    gini=gini,
                    entropy=entropy,
                ),
                "sequence_context_profile": self._sequence_context_profile(
                    avg_len=interactions / users,
                    p90_len=float(user_counts.quantile(0.90)),
                    repeat_share=repeat_item_share,
                ),
            }

            if timestamp_col:
                timestamps = pd.to_numeric(data[timestamp_col], errors="coerce")
                span_days = (timestamps.max() - timestamps.min()) / 86400
                row["time_span_days"] = span_days
                row["temporal_profile"] = self._temporal_profile(span_days)

            rows.append(row)

        return pd.DataFrame(rows).sort_values("dataset")

    def popular_items(self, datasets: list[str], top_k: int = 25) -> pd.DataFrame:
        rows = []
        for dataset in datasets:
            data = self._load_interactions(dataset)
            item_col = self._find_column(data, "item_id")
            timestamp_col = self._find_column(data, "timestamp")
            item_counts = data[item_col].value_counts()
            total = len(data)

            if timestamp_col:
                timestamps = pd.to_numeric(data[timestamp_col], errors="coerce")
                cutoff = timestamps.quantile(0.90)
                recent_counts = data.loc[timestamps >= cutoff, item_col].value_counts()
            else:
                recent_counts = pd.Series(dtype="int64")

            for rank, (item_id, count) in enumerate(item_counts.head(top_k).items(), 1):
                recent_count = int(recent_counts.get(item_id, 0))
                rows.append(
                    {
                        "dataset": dataset,
                        "rank": rank,
                        "item_id": item_id,
                        "interaction_count": int(count),
                        "interaction_share": count / total,
                        "recent_top10_percent_count": recent_count,
                        "recent_top10_percent_share_within_item": (
                            recent_count / count if count else 0.0
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def model_explanation_summary(
        self,
        comparative_summary: pd.DataFrame,
        dataset_behavior: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = comparative_summary.merge(
            dataset_behavior[
                [
                    "dataset",
                    "top10_item_share",
                    "top1_percent_item_share",
                    "item_popularity_gini",
                    "item_popularity_normalized_entropy",
                    "avg_interactions_per_user_or_session",
                    "repeat_item_share_within_user_or_session",
                    "sequence_context_profile",
                    "popularity_profile",
                ]
            ],
            on="dataset",
            how="left",
        )

        frame["interpretation"] = frame.apply(self._interpret_model_row, axis=1)
        return frame[
            [
                "experiment_type",
                "dataset",
                "model",
                "mrr@10",
                "relative_mrr@10_to_dataset_best",
                "hit@10",
                "ndcg@10",
                "runtime_seconds",
                "top10_item_share",
                "top1_percent_item_share",
                "item_popularity_gini",
                "avg_interactions_per_user_or_session",
                "repeat_item_share_within_user_or_session",
                "popularity_profile",
                "sequence_context_profile",
                "interpretation",
            ]
        ].sort_values(["experiment_type", "dataset", "mrr@10"], ascending=[True, True, False])

    def knn_vstan_explanation(
        self,
        best_per_model: pd.DataFrame,
        popularity_weighting: pd.DataFrame,
        dataset_behavior: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        behavior = dataset_behavior.set_index("dataset")
        best = best_per_model[
            best_per_model["model"].isin(SESSION_NEIGHBORHOOD_MODELS)
        ].copy()

        for _, row in best.iterrows():
            dataset = row["dataset"]
            model = row["model"]
            dataset_stats = behavior.loc[dataset]
            weight_row = popularity_weighting[
                (popularity_weighting["dataset"] == dataset)
                & (popularity_weighting["model"] == model)
            ]
            weight_delta = (
                float(weight_row.iloc[0]["delta_mrr@10_weighted_minus_unweighted"])
                if not weight_row.empty
                else pd.NA
            )

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mrr@10": row.get("mrr@10"),
                    "hit@10": row.get("hit@10"),
                    "ndcg@10": row.get("ndcg@10"),
                    "runtime_seconds": row.get("runtime_seconds"),
                    "best_config": row.get("config_json"),
                    "top10_item_share": dataset_stats["top10_item_share"],
                    "top1_percent_item_share": dataset_stats[
                        "top1_percent_item_share"
                    ],
                    "item_popularity_gini": dataset_stats["item_popularity_gini"],
                    "avg_session_length": dataset_stats[
                        "avg_interactions_per_user_or_session"
                    ],
                    "p90_session_length": dataset_stats[
                        "p90_interactions_per_user_or_session"
                    ],
                    "repeat_item_share_within_session": dataset_stats[
                        "repeat_item_share_within_user_or_session"
                    ],
                    "popularity_weighting_delta_mrr@10": weight_delta,
                    "explanation": self._interpret_knn_vstan(
                        model=model,
                        dataset_stats=dataset_stats,
                        weight_delta=weight_delta,
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(["dataset", "model"])

    def popularity_baseline_explanation(
        self,
        comparative_summary: pd.DataFrame,
        dataset_behavior: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = comparative_summary[
            comparative_summary["model"].isin(POPULARITY_MODELS)
        ].merge(
            dataset_behavior[
                [
                    "dataset",
                    "top1_item_share",
                    "top10_item_share",
                    "top100_item_share",
                    "top1_percent_item_share",
                    "item_popularity_gini",
                    "item_popularity_normalized_entropy",
                    "time_span_days",
                    "popularity_profile",
                    "temporal_profile",
                ]
            ],
            on="dataset",
            how="left",
        )
        frame["explanation"] = frame.apply(self._interpret_popularity_baseline, axis=1)
        return frame[
            [
                "experiment_type",
                "dataset",
                "model",
                "mrr@10",
                "relative_mrr@10_to_dataset_best",
                "hit@10",
                "ndcg@10",
                "coverage@10",
                "avg_recommendation_popularity@10",
                "top1_item_share",
                "top10_item_share",
                "top100_item_share",
                "top1_percent_item_share",
                "item_popularity_gini",
                "item_popularity_normalized_entropy",
                "time_span_days",
                "popularity_profile",
                "temporal_profile",
                "explanation",
            ]
        ].sort_values(["experiment_type", "dataset", "mrr@10"], ascending=[True, True, False])

    def write_outputs(self, **tables: pd.DataFrame) -> dict[str, Path]:
        paths = {}
        for name, table in tables.items():
            path = self.output_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path
        return paths

    def create_plots(
        self,
        dataset_behavior: pd.DataFrame,
        popular_items: pd.DataFrame,
    ) -> dict[str, Path]:
        if plt is None:
            return {}

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        path = plot_dir / "popularity_concentration_by_dataset.png"
        self._plot_popularity_concentration(dataset_behavior, path)
        paths["plot_popularity_concentration"] = path

        path = plot_dir / "top_item_share_by_dataset.png"
        self._plot_top_item_shares(dataset_behavior, path)
        paths["plot_top_item_shares"] = path

        path = plot_dir / "top25_item_popularity_curves.png"
        self._plot_top_item_curves(popular_items, path)
        paths["plot_top25_item_popularity_curves"] = path

        return paths

    def write_report(
        self,
        dataset_behavior: pd.DataFrame,
        model_explanation: pd.DataFrame,
        knn_vstan_explanation: pd.DataFrame,
        pop_baseline_explanation: pd.DataFrame,
        plot_paths: dict[str, Path],
    ) -> Path:
        path = self.output_dir / "model_behavior_explanation.md"
        lines = [
            "# Model Behavior Explanation",
            "",
            "This report connects dataset characteristics with model performance. "
            "It is intended to support the thesis discussion around why some "
            "models work better on some datasets and worse on others.",
            "",
            "## Reading Guide",
            "",
            "- High `top10_item_share`, high `top1_percent_item_share`, and high "
            "`item_popularity_gini` indicate that interactions are concentrated "
            "on a small set of popular items.",
            "- Popularity baselines are expected to work better when the next "
            "interaction is often one of these highly popular items.",
            "- VS-KNN and VSTAN are expected to work better when sessions contain "
            "usable co-occurrence or transition signals.",
            "- If inverse popularity weighting hurts VS-KNN or VSTAN, this "
            "suggests that popular items carry useful predictive signal rather "
            "than being only a bias artifact.",
            "",
            "## Dataset Behavior Summary",
            "",
            self._to_markdown(dataset_behavior),
            "",
            "## VS-KNN and VSTAN Explanation",
            "",
            self._to_markdown(knn_vstan_explanation),
            "",
            "## Popularity Baseline Explanation",
            "",
            self._to_markdown(pop_baseline_explanation),
            "",
            "## Model-Level Explanation",
            "",
            self._to_markdown(model_explanation),
            "",
        ]

        if plot_paths:
            lines.extend(["## Plots", ""])
            for name, plot_path in plot_paths.items():
                title = name.replace("plot_", "").replace("_", " ").title()
                relative = plot_path.relative_to(self.output_dir).as_posix()
                lines.extend([f"### {title}", "", f"![{title}]({relative})", ""])

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.structured_report_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _load_interactions(self, dataset: str) -> pd.DataFrame:
        path = self.project_root / "data" / "recbole" / dataset / f"{dataset}.inter"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path, sep="\t")

    @staticmethod
    def _find_column(data: pd.DataFrame, logical_name: str) -> str:
        for column in data.columns:
            if column.split(":", maxsplit=1)[0] == logical_name:
                return column
        raise KeyError(f"Could not find RecBole column {logical_name!r}")

    @staticmethod
    def _experiment_type(dataset: str) -> str:
        lowered = dataset.lower()
        if any(marker in lowered for marker in SESSION_DATASET_MARKERS):
            return "session"
        return "topn"

    @staticmethod
    def _top_share(counts: pd.Series, n: int) -> float:
        if counts.empty:
            return 0.0
        return float(counts.head(n).sum() / counts.sum())

    @staticmethod
    def _top_fraction_share(counts: pd.Series, fraction: float) -> float:
        if counts.empty:
            return 0.0
        n = max(1, int(round(len(counts) * fraction)))
        return float(counts.head(n).sum() / counts.sum())

    @staticmethod
    def _gini(counts: pd.Series) -> float:
        values = counts.astype(float).sort_values().to_numpy()
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        n = len(values)
        index = pd.Series(range(1, n + 1), dtype=float).to_numpy()
        return float(((2 * index - n - 1) * values).sum() / (n * values.sum()))

    @staticmethod
    def _normalized_entropy(counts: pd.Series) -> float:
        if counts.empty:
            return 0.0
        probabilities = counts.astype(float) / counts.sum()
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        max_entropy = math.log(len(probabilities))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def _repeat_item_share(data: pd.DataFrame, user_col: str, item_col: str) -> float:
        repeated = data.duplicated(subset=[user_col, item_col], keep="first")
        return float(repeated.mean()) if len(repeated) else 0.0

    @staticmethod
    def _popularity_profile(
        top10_share: float,
        top1pct_share: float,
        gini: float,
        entropy: float,
    ) -> str:
        if top10_share >= 0.25 or top1pct_share >= 0.55 or gini >= 0.85:
            return "highly concentrated popularity"
        if top10_share >= 0.10 or top1pct_share >= 0.35 or gini >= 0.70:
            return "moderately concentrated popularity"
        if entropy >= 0.85:
            return "diffuse popularity"
        return "mixed popularity profile"

    @staticmethod
    def _sequence_context_profile(
        avg_len: float,
        p90_len: float,
        repeat_share: float,
    ) -> str:
        if avg_len >= 20:
            return "long user histories"
        if avg_len >= 4 or p90_len >= 8:
            return "moderate session context"
        if repeat_share >= 0.10:
            return "short sessions with repeated items"
        return "very short session context"

    @staticmethod
    def _temporal_profile(span_days: float) -> str:
        if span_days <= 3:
            return "very short time span"
        if span_days <= 45:
            return "short to medium time span"
        if span_days <= 365:
            return "medium time span"
        return "long time span"

    @staticmethod
    def _interpret_model_row(row: pd.Series) -> str:
        model = row["model"]
        popularity = row["popularity_profile"]
        context = row["sequence_context_profile"]
        relative = row.get("relative_mrr@10_to_dataset_best", pd.NA)

        if model in POPULARITY_MODELS:
            if "concentrated" in popularity and relative >= 0.5:
                return "Popularity is a meaningful signal on this dataset; the model benefits from item concentration."
            if relative < 0.2:
                return "Global popularity does not explain the next interaction well; session or personalization signals dominate."
            return "Popularity carries some signal, but it is not sufficient to match the strongest model."

        if model in SESSION_NEIGHBORHOOD_MODELS:
            if "moderate" in context or "long" in context:
                return "Session-neighborhood information is usable; similar sessions provide predictive item candidates."
            return "The model can still exploit co-occurrences, but very short sessions limit neighborhood evidence."

        if model == "GRU4Rec":
            if relative >= 0.9:
                return "The neural sequence model captures useful transition patterns for this dataset."
            return "The neural model is competitive but not clearly dominant, suggesting that simpler sequence neighborhoods may capture enough signal."

        if model == "BPR":
            return "The latent factor model benefits from repeated user-item history and personalization signal."

        return "Model behavior should be interpreted together with dataset structure and runtime."

    @staticmethod
    def _interpret_knn_vstan(
        model: str,
        dataset_stats: pd.Series,
        weight_delta: float | pd.NA,
    ) -> str:
        context = dataset_stats["sequence_context_profile"]
        popularity = dataset_stats["popularity_profile"]
        top10 = dataset_stats["top10_item_share"]
        repeat_share = dataset_stats["repeat_item_share_within_user_or_session"]

        parts = []
        if model == "VSTAN":
            parts.append("VSTAN can use recency and position effects in addition to session-neighborhood overlap.")
        else:
            parts.append("VS-KNN relies mainly on overlap with similar historical sessions.")

        if "very short" in context:
            parts.append("The short sessions limit how much neighborhood evidence is available.")
        else:
            parts.append("The available session context gives the neighborhood method useful co-occurrence evidence.")

        if "concentrated" in popularity or top10 >= 0.10:
            parts.append("Popular items are predictive in this dataset.")
        else:
            parts.append("Popularity is more diffuse, so item-neighborhood structure matters more than a few dominant items.")

        if pd.notna(weight_delta) and weight_delta < -0.02:
            parts.append("Inverse popularity weighting strongly hurts, which indicates that down-weighting popular items removes useful signal.")
        elif pd.notna(weight_delta) and weight_delta < 0:
            parts.append("Inverse popularity weighting slightly hurts, so popularity still carries useful signal but is less dominant.")

        if repeat_share >= 0.10:
            parts.append("Repeated items inside sessions may make simple popularity or repeat-aware signals unusually strong.")

        return " ".join(parts)

    @staticmethod
    def _interpret_popularity_baseline(row: pd.Series) -> str:
        model = row["model"]
        relative = row.get("relative_mrr@10_to_dataset_best", pd.NA)
        popularity = row.get("popularity_profile", "")
        temporal = row.get("temporal_profile", "")

        if relative >= 0.5:
            base = "The popularity baseline explains a large part of the best observed performance."
        elif relative >= 0.2:
            base = "The popularity baseline is useful but clearly incomplete."
        else:
            base = "The popularity baseline is weak, indicating that next-item behavior is not explained by global popularity alone."

        if model == "RecentPop":
            detail = "RecentPop depends on whether the selected time window matches the dataset's temporal dynamics."
        elif model == "DecayPop":
            detail = "DecayPop can help when older interactions should gradually lose influence."
        else:
            detail = "MostPop works best when long-term popularity is stable and strongly concentrated."

        return f"{base} {detail} Dataset profile: {popularity}; {temporal}."

    @staticmethod
    def _plot_popularity_concentration(
        dataset_behavior: pd.DataFrame,
        path: Path,
    ) -> None:
        plot = dataset_behavior.set_index("dataset")[
            ["item_popularity_gini", "item_popularity_normalized_entropy"]
        ]
        plot.plot(kind="bar", figsize=(10, 5), color=["#3B6EA8", "#C45A4A"])
        plt.ylabel("score")
        plt.title("Item popularity concentration by dataset")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _plot_top_item_shares(dataset_behavior: pd.DataFrame, path: Path) -> None:
        plot = dataset_behavior.set_index("dataset")[
            ["top1_item_share", "top10_item_share", "top100_item_share"]
        ]
        plot.plot(kind="bar", figsize=(10, 5))
        plt.ylabel("share of interactions")
        plt.title("Interaction share captured by top popular items")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _plot_top_item_curves(popular_items: pd.DataFrame, path: Path) -> None:
        plt.figure(figsize=(10, 5))
        for dataset, group in popular_items.groupby("dataset"):
            plt.plot(
                group["rank"],
                group["interaction_share"],
                marker="o",
                linewidth=1.5,
                label=dataset,
            )
        plt.xlabel("popular item rank")
        plt.ylabel("interaction share")
        plt.title("Top-25 item popularity curves")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _to_markdown(table: pd.DataFrame) -> str:
        if table.empty:
            return "No data available."

        rounded = table.copy()
        for column in rounded.select_dtypes(include=["float"]).columns:
            rounded[column] = rounded[column].round(4)
        text_table = rounded.fillna("").astype(str)

        headers = list(text_table.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in text_table.values.tolist():
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain model behavior using dataset popularity diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explainer = ModelBehaviorExplainer(output_dir=args.output_dir)
    paths = explainer.run()
    print("Model behavior explanation written to:")
    for name, path in sorted(paths.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
