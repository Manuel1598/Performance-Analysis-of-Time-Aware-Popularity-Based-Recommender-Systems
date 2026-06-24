from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

METRIC_COLUMNS = [
    "hit@5",
    "hit@10",
    "ndcg@5",
    "ndcg@10",
    "mrr@5",
    "mrr@10",
    "coverage@10",
    "avg_recommendation_popularity@10",
]

RUNTIME_COLUMNS = [
    "runtime_seconds",
    "train_runtime_seconds",
    "eval_runtime_seconds",
    "extra_metrics_runtime_seconds",
]

DERIVED_COLUMNS = [
    "runtime_minutes",
    "train_runtime_share",
    "eval_runtime_share",
    "extra_metrics_runtime_share",
    "mrr@10_per_minute",
    "ndcg@10_per_minute",
    "hit@10_per_minute",
    "rank_by_mrr@10",
    "relative_mrr@10_to_dataset_best",
    "mrr@10_gap_to_dataset_best",
    "runtime_relative_to_dataset_fastest",
    "quality_runtime_pareto_efficient",
]

BASE_COLUMNS = [
    "experiment_type",
    "dataset",
    "model",
    "device",
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "status",
]

KNOWN_CONFIG_COLUMNS = [
    "window_days",
    "decay_lambda",
    "embedding_size",
    "learning_rate",
    "train_neg_sample_args",
    "vsknn_k",
    "vsknn_sample_size",
    "vsknn_popularity_weight",
    "vstan_k",
    "vstan_sample_size",
    "vstan_position_decay",
    "vstan_idf_weighting",
    "vstan_popularity_weight",
    "hidden_size",
    "dropout_prob",
    "num_layers",
    "loss_type",
]

MODEL_FAMILIES = {
    "MostPop": "popularity baseline",
    "RecentPop": "time-aware popularity",
    "DecayPop": "time-aware popularity",
    "BPR": "latent factor baseline",
    "NeuMF": "neural baseline",
    "SVD": "latent factor baseline",
    "VS-KNN": "session neighborhood",
    "VSTAN": "time-aware session neighborhood",
    "GRU4Rec": "neural session model",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    result_file: Path
    log_file: Path | None = None


class RecboleResultEvaluator:
    """Create structured reports for RecBole Top-N and session experiments."""

    def __init__(
        self,
        project_root: Path | None = None,
        results_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.results_root = results_root or self.project_root / "recbole_results"
        self.output_dir = (
            output_dir
            or self.results_root / "tuning_results" / "analysis_results"
            / "structured_report"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def full_experiments(self) -> list[ExperimentSpec]:
        return [
            ExperimentSpec(
                name="session",
                result_file=(
                    self.results_root
                    / "tuning_results"
                    / "session_full_tuning_results.csv"
                ),
                log_file=(
                    self.results_root
                    / "experiment_logs"
                    / "session_full_tuning_experiment_log.csv"
                ),
            ),
            ExperimentSpec(
                name="topn",
                result_file=(
                    self.results_root
                    / "tuning_results"
                    / "topn_full_tuning_results.csv"
                ),
                log_file=(
                    self.results_root
                    / "experiment_logs"
                    / "topn_full_tuning_experiment_log.csv"
                ),
            ),
        ]

    def all_experiments(self) -> list[ExperimentSpec]:
        experiments: list[ExperimentSpec] = []
        for result_file in sorted(
            (self.results_root / "tuning_results").glob("*_tuning_results.csv")
        ):
            name = result_file.name.removesuffix("_tuning_results.csv")
            log_file = (
                self.results_root
                / "experiment_logs"
                / f"{name}_tuning_experiment_log.csv"
            )
            experiments.append(
                ExperimentSpec(
                    name=name,
                    result_file=result_file,
                    log_file=log_file if log_file.exists() else None,
                )
            )
        return experiments

    def evaluate(
        self,
        experiments: Iterable[ExperimentSpec] | None = None,
    ) -> dict[str, Path]:
        specs = list(experiments or self.full_experiments())
        results = self.load_results(specs)
        results = self._drop_legacy_session_knn_rows(results)
        successful = self.successful_results(results)
        dataset_summary = self.dataset_summary(successful["dataset"].unique())
        model_summary = self.model_summary(results)
        tuning_summary = self.tuning_summary(results)
        best_overall = self.best_overall(successful)
        best_per_model = self.best_per_model(successful)
        runtime_summary = self.runtime_summary(successful)
        efficiency_summary = self.efficiency_summary(successful)
        popularity_weighting_summary = self.popularity_weighting_summary(successful)
        comparative_summary = self.comparative_summary(best_per_model)

        paths = self.write_tables(
            dataset_summary=dataset_summary,
            model_summary=model_summary,
            tuning_summary=tuning_summary,
            best_overall=best_overall,
            best_per_model=best_per_model,
            runtime_summary=runtime_summary,
            efficiency_summary=efficiency_summary,
            popularity_weighting_summary=popularity_weighting_summary,
            comparative_summary=comparative_summary,
        )
        plot_paths = self.create_plots(successful, best_per_model)
        report_path = self.write_markdown_report(
            results=results,
            dataset_summary=dataset_summary,
            model_summary=model_summary,
            tuning_summary=tuning_summary,
            best_overall=best_overall,
            best_per_model=best_per_model,
            runtime_summary=runtime_summary,
            efficiency_summary=efficiency_summary,
            popularity_weighting_summary=popularity_weighting_summary,
            comparative_summary=comparative_summary,
            plot_paths=plot_paths,
        )
        paths.update(plot_paths)
        paths["report"] = report_path
        return paths

    def load_results(self, specs: Iterable[ExperimentSpec]) -> pd.DataFrame:
        frames = []
        for spec in specs:
            if not spec.result_file.exists():
                continue

            frame = pd.read_csv(spec.result_file)
            frame["experiment_type"] = self._experiment_type(spec.name)
            frame["experiment_name"] = spec.name
            frame["source_file"] = str(spec.result_file)
            frame["has_experiment_log"] = bool(spec.log_file and spec.log_file.exists())
            frames.append(frame)

        if not frames:
            raise FileNotFoundError(
                f"No tuning result CSV files found in {self.results_root}"
            )

        results = pd.concat(frames, ignore_index=True, sort=False)
        results = self._clean_result_rows(results)
        for column in METRIC_COLUMNS + RUNTIME_COLUMNS:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        return self.add_derived_metrics(results)

    @staticmethod
    def _drop_legacy_session_knn_rows(results: pd.DataFrame) -> pd.DataFrame:
        """Ignore stale session KNN rows when newer weighted-grid rows exist."""
        frame = results.copy()
        legacy_mask = pd.Series(False, index=frame.index)
        model_weight_columns = {
            "VS-KNN": "vsknn_popularity_weight",
            "VSTAN": "vstan_popularity_weight",
        }

        for model, weight_column in model_weight_columns.items():
            if weight_column not in frame.columns:
                continue

            model_mask = (
                frame["experiment_type"].eq("session")
                & frame["model"].eq(model)
            )
            for _, group in frame[model_mask].groupby("dataset", dropna=False):
                if group[weight_column].notna().any():
                    legacy_mask.loc[group[group[weight_column].isna()].index] = True

        return frame.loc[~legacy_mask].reset_index(drop=True)

    @staticmethod
    def add_derived_metrics(results: pd.DataFrame) -> pd.DataFrame:
        frame = results.copy()

        if "runtime_seconds" in frame.columns:
            positive_runtime = frame["runtime_seconds"].where(
                frame["runtime_seconds"] > 0
            )
            frame["runtime_minutes"] = frame["runtime_seconds"] / 60

            for metric in ["mrr@10", "ndcg@10", "hit@10"]:
                if metric in frame.columns:
                    frame[f"{metric}_per_minute"] = (
                        frame[metric]
                        / frame["runtime_minutes"].where(
                            frame["runtime_minutes"] > 0
                        )
                    )

            for column in [
                "train_runtime_seconds",
                "eval_runtime_seconds",
                "extra_metrics_runtime_seconds",
            ]:
                if column in frame.columns:
                    share_column = column.replace("_seconds", "_share")
                    frame[share_column] = frame[column] / positive_runtime

        if {"experiment_type", "dataset", "mrr@10"}.issubset(frame.columns):
            group_keys = ["experiment_type", "dataset"]
            dataset_best = frame.groupby(group_keys)["mrr@10"].transform("max")
            frame["rank_by_mrr@10"] = frame.groupby(group_keys)["mrr@10"].rank(
                method="min",
                ascending=False,
            )
            frame["relative_mrr@10_to_dataset_best"] = (
                frame["mrr@10"] / dataset_best.where(dataset_best > 0)
            )
            frame["mrr@10_gap_to_dataset_best"] = dataset_best - frame["mrr@10"]

        if {"experiment_type", "dataset", "runtime_seconds"}.issubset(frame.columns):
            group_keys = ["experiment_type", "dataset"]
            fastest = frame.groupby(group_keys)["runtime_seconds"].transform("min")
            frame["runtime_relative_to_dataset_fastest"] = (
                frame["runtime_seconds"] / fastest.where(fastest > 0)
            )

        if {"experiment_type", "dataset", "mrr@10", "runtime_seconds"}.issubset(
            frame.columns
        ):
            frame["quality_runtime_pareto_efficient"] = False
            for _, group in frame.groupby(["experiment_type", "dataset"]):
                pareto_mask = RecboleResultEvaluator._pareto_front(
                    quality=group["mrr@10"],
                    cost=group["runtime_seconds"],
                )
                frame.loc[group.index, "quality_runtime_pareto_efficient"] = (
                    pareto_mask
                )

        return frame

    @staticmethod
    def successful_results(results: pd.DataFrame) -> pd.DataFrame:
        if "status" not in results.columns:
            return results.copy()
        return results[results["status"].fillna("") == "success"].copy()

    def dataset_summary(self, dataset_names: Iterable[str]) -> pd.DataFrame:
        rows = []
        for dataset in sorted(str(name) for name in dataset_names if pd.notna(name)):
            inter_file = (
                self.project_root
                / "data"
                / "recbole"
                / dataset
                / f"{dataset}.inter"
            )
            if not inter_file.exists():
                rows.append({"dataset": dataset, "inter_file_found": False})
                continue

            data = pd.read_csv(inter_file, sep="\t")
            user_col = self._find_column(data.columns, "user_id")
            item_col = self._find_column(data.columns, "item_id")
            timestamp_col = self._find_column(data.columns, "timestamp")

            interactions = len(data)
            users = data[user_col].nunique() if user_col else pd.NA
            items = data[item_col].nunique() if item_col else pd.NA
            density = (
                interactions / (users * items)
                if user_col and item_col and users and items
                else pd.NA
            )

            row = {
                "dataset": dataset,
                "inter_file_found": True,
                "interactions": interactions,
                "users_or_sessions": users,
                "items": items,
                "avg_interactions_per_user_or_session": (
                    interactions / users if user_col and users else pd.NA
                ),
                "avg_interactions_per_item": (
                    interactions / items if item_col and items else pd.NA
                ),
                "interaction_matrix_density": density,
            }

            if timestamp_col:
                timestamps = pd.to_numeric(data[timestamp_col], errors="coerce")
                row["timestamp_min"] = self._format_timestamp(timestamps.min())
                row["timestamp_max"] = self._format_timestamp(timestamps.max())

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def model_summary(results: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = results.groupby(["experiment_type", "dataset", "model"], dropna=False)
        for keys, group in grouped:
            experiment_type, dataset, model = keys
            successful_runs = int((group.get("status") == "success").sum())
            rows.append(
                {
                    "experiment_type": experiment_type,
                    "dataset": dataset,
                    "model": model,
                    "model_family": MODEL_FAMILIES.get(str(model), "other"),
                    "runs": len(group),
                    "successful_runs": successful_runs,
                    "failed_runs": len(group) - successful_runs,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def tuning_summary(results: pd.DataFrame) -> pd.DataFrame:
        rows = []
        available_config_columns = [
            column for column in KNOWN_CONFIG_COLUMNS if column in results.columns
        ]

        grouped = results.groupby(["experiment_type", "dataset", "model"], dropna=False)
        for keys, group in grouped:
            experiment_type, dataset, model = keys
            row = {
                "experiment_type": experiment_type,
                "dataset": dataset,
                "model": model,
                "runs": len(group),
                "unique_configurations": RecboleResultEvaluator._unique_configs(group),
            }
            for column in available_config_columns:
                values = group[column].dropna().unique().tolist()
                if values:
                    row[column] = RecboleResultEvaluator._compact_values(values)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def best_overall(results: pd.DataFrame, metric: str = "mrr@10") -> pd.DataFrame:
        if metric not in results.columns:
            metric = RecboleResultEvaluator._first_available_metric(results)
        return results.sort_values(metric, ascending=False).head(30)

    @staticmethod
    def best_per_model(results: pd.DataFrame, metric: str = "mrr@10") -> pd.DataFrame:
        if metric not in results.columns:
            metric = RecboleResultEvaluator._first_available_metric(results)
        return (
            results.sort_values(metric, ascending=False)
            .groupby(["experiment_type", "dataset", "model"], as_index=False)
            .first()
        )

    @staticmethod
    def runtime_summary(results: pd.DataFrame) -> pd.DataFrame:
        runtime_cols = [col for col in RUNTIME_COLUMNS if col in results.columns]
        if not runtime_cols:
            return pd.DataFrame()
        summary = (
            results.groupby(["experiment_type", "dataset", "model"], dropna=False)[
                runtime_cols
            ]
            .agg(["count", "mean", "median", "min", "max"])
            .round(4)
        ).reset_index()
        summary.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        return summary

    @staticmethod
    def efficiency_summary(results: pd.DataFrame) -> pd.DataFrame:
        metric = (
            "mrr@10"
            if "mrr@10" in results.columns
            else RecboleResultEvaluator._first_available_metric(results)
        )
        if "runtime_seconds" not in results.columns:
            return pd.DataFrame()

        frame = results.copy()
        frame["primary_metric"] = frame[metric]
        frame["primary_metric_name"] = metric
        frame["primary_metric_per_runtime_second"] = (
            frame[metric] / frame["runtime_seconds"].where(frame["runtime_seconds"] > 0)
        )
        return (
            frame.sort_values(
                "primary_metric_per_runtime_second",
                ascending=False,
            )
            .groupby(["experiment_type", "dataset", "model"], as_index=False)
            .first()
        )

    @staticmethod
    def popularity_weighting_summary(
        results: pd.DataFrame,
        metric: str = "mrr@10",
    ) -> pd.DataFrame:
        rows = []

        model_weight_columns = {
            "VS-KNN": "vsknn_popularity_weight",
            "VSTAN": "vstan_popularity_weight",
        }

        for model, weight_column in model_weight_columns.items():
            if weight_column not in results.columns:
                continue

            model_results = results[results["model"] == model].copy()

            if model_results.empty:
                continue

            model_results[weight_column] = pd.to_numeric(
                model_results[weight_column],
                errors="coerce",
            )

            for keys, group in model_results.groupby(
                ["experiment_type", "dataset"],
                dropna=False,
            ):
                experiment_type, dataset = keys

                explicit = group[group[weight_column].notna()].copy()
                unweighted = explicit[explicit[weight_column] == 0.0]
                weighted = explicit[explicit[weight_column] > 0.0]

                if unweighted.empty:
                    unweighted = group[group[weight_column].isna()]

                if unweighted.empty or weighted.empty:
                    continue

                best_unweighted = unweighted.sort_values(
                    metric,
                    ascending=False,
                ).iloc[0]
                best_weighted = weighted.sort_values(
                    metric,
                    ascending=False,
                ).iloc[0]

                row = {
                    "experiment_type": experiment_type,
                    "dataset": dataset,
                    "model": model,
                    "best_unweighted_mrr@10": best_unweighted.get("mrr@10"),
                    "best_weighted_mrr@10": best_weighted.get("mrr@10"),
                    "delta_mrr@10_weighted_minus_unweighted": (
                        best_weighted.get("mrr@10")
                        - best_unweighted.get("mrr@10")
                    ),
                    "weighted_improves_mrr@10": (
                        best_weighted.get("mrr@10")
                        > best_unweighted.get("mrr@10")
                    ),
                    "best_unweighted_hit@10": best_unweighted.get("hit@10"),
                    "best_weighted_hit@10": best_weighted.get("hit@10"),
                    "delta_hit@10_weighted_minus_unweighted": (
                        best_weighted.get("hit@10")
                        - best_unweighted.get("hit@10")
                    ),
                    "best_unweighted_ndcg@10": best_unweighted.get("ndcg@10"),
                    "best_weighted_ndcg@10": best_weighted.get("ndcg@10"),
                    "delta_ndcg@10_weighted_minus_unweighted": (
                        best_weighted.get("ndcg@10")
                        - best_unweighted.get("ndcg@10")
                    ),
                    "best_weighted_popularity_weight": best_weighted.get(
                        weight_column
                    ),
                    "best_unweighted_runtime_seconds": best_unweighted.get(
                        "runtime_seconds"
                    ),
                    "best_weighted_runtime_seconds": best_weighted.get(
                        "runtime_seconds"
                    ),
                    "best_unweighted_config": best_unweighted.get(
                        "config_json"
                    ),
                    "best_weighted_config": best_weighted.get("config_json"),
                }

                rows.append(row)

        summary = pd.DataFrame(rows)

        if not summary.empty:
            summary = summary.sort_values(["experiment_type", "dataset", "model"])

        return summary

    @staticmethod
    def comparative_summary(best_per_model: pd.DataFrame) -> pd.DataFrame:
        desired_columns = [
            "experiment_type",
            "dataset",
            "model",
            "mrr@10",
            "rank_by_mrr@10",
            "relative_mrr@10_to_dataset_best",
            "mrr@10_gap_to_dataset_best",
            "hit@10",
            "ndcg@10",
            "coverage@10",
            "avg_recommendation_popularity@10",
            "runtime_seconds",
            "runtime_minutes",
            "runtime_relative_to_dataset_fastest",
            "mrr@10_per_minute",
            "train_runtime_share",
            "eval_runtime_share",
            "extra_metrics_runtime_share",
            "quality_runtime_pareto_efficient",
        ]
        columns = [column for column in desired_columns if column in best_per_model]
        summary = best_per_model[columns].copy()
        sort_columns = [
            column
            for column in ["experiment_type", "dataset", "rank_by_mrr@10"]
            if column in summary
        ]
        if sort_columns:
            summary = summary.sort_values(sort_columns)
        return summary

    def write_tables(self, **tables: pd.DataFrame) -> dict[str, Path]:
        paths = {}
        for name, table in tables.items():
            path = self.output_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path
        return paths

    def create_plots(
        self,
        results: pd.DataFrame,
        best_per_model: pd.DataFrame,
    ) -> dict[str, Path]:
        if plt is None:
            return {}

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        metric = (
            "mrr@10"
            if "mrr@10" in results.columns
            else self._first_available_metric(results)
        )
        if metric in best_per_model.columns:
            path = plot_dir / "best_metric_per_model.png"
            self._plot_best_metric(best_per_model, metric, path)
            paths["plot_best_metric_per_model"] = path

        if "runtime_seconds" in best_per_model.columns and metric in best_per_model:
            path = plot_dir / "runtime_vs_metric.png"
            self._plot_runtime_vs_metric(best_per_model, metric, path)
            paths["plot_runtime_vs_metric"] = path

        if "runtime_seconds" in results.columns:
            path = plot_dir / "runtime_distribution.png"
            self._plot_runtime_distribution(results, path)
            paths["plot_runtime_distribution"] = path

        return paths

    def write_markdown_report(
        self,
        results: pd.DataFrame,
        dataset_summary: pd.DataFrame,
        model_summary: pd.DataFrame,
        tuning_summary: pd.DataFrame,
        best_overall: pd.DataFrame,
        best_per_model: pd.DataFrame,
        runtime_summary: pd.DataFrame,
        efficiency_summary: pd.DataFrame,
        popularity_weighting_summary: pd.DataFrame,
        comparative_summary: pd.DataFrame,
        plot_paths: dict[str, Path],
    ) -> Path:
        metric = (
            "mrr@10"
            if "mrr@10" in results.columns
            else self._first_available_metric(results)
        )
        report_path = self.output_dir / "recbole_structured_evaluation.md"

        lines = [
            "# Structured RecBole Evaluation",
            "",
            "This report summarizes the available RecBole tuning results for "
            "Top-N and session-based recommendation experiments.",
            "",
            "## Scope",
            "",
            f"- Result rows: {len(results)}",
            f"- Successful rows: {len(self.successful_results(results))}",
            f"- Datasets: {', '.join(sorted(results['dataset'].dropna().unique()))}",
            f"- Models: {', '.join(sorted(results['model'].dropna().unique()))}",
            f"- Primary ranking metric: `{metric}`",
            "",
        ]

        lines.extend(self.metric_guidance())
        self._append_section(lines, "Dataset Characteristics", dataset_summary)
        self._append_section(lines, "Models and Run Counts", model_summary)
        self._append_section(lines, "Tuning Search Space", tuning_summary)
        self._append_section(lines, "Best Results Overall", best_overall)
        self._append_section(lines, "Best Configuration per Model", best_per_model)
        self._append_section(
            lines,
            "Quality and Runtime Comparison",
            comparative_summary,
        )
        self._append_section(lines, "Runtime Summary", runtime_summary)
        self._append_section(lines, "Efficiency Summary", efficiency_summary)
        self._append_section(
            lines,
            "Popularity Weighting Comparison",
            popularity_weighting_summary,
        )

        if plot_paths:
            lines.extend(["## Plots", ""])
            for name, path in plot_paths.items():
                relative = path.relative_to(self.output_dir).as_posix()
                title = name.replace("plot_", "").replace("_", " ").title()
                lines.extend([f"### {title}", "", f"![{title}]({relative})", ""])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    @staticmethod
    def metric_guidance() -> list[str]:
        return [
            "## Recommended Reading of Metrics",
            "",
            "- `MRR@10` is the main quality metric here because it rewards models "
            "that place the first relevant item very high. This is especially "
            "important for session recommendation.",
            "- `NDCG@10` is the best supporting ranking metric because it rewards "
            "multiple relevant hits while still valuing top positions more.",
            "- `Hit@10` is useful as an intuitive recall-style metric, but it does "
            "not distinguish rank 1 from rank 10.",
            "- `Coverage@10` and `avg_recommendation_popularity@10` are important "
            "for the thesis question around popularity bias. High accuracy with "
            "very low coverage can mean the model recommends a narrow popular "
            "item set.",
            "- `runtime_seconds`, runtime shares, and `mrr@10_per_minute` show "
            "whether a quality gain is computationally worth it.",
            "- `quality_runtime_pareto_efficient` marks configurations that are "
            "not dominated by another configuration with both better/equal "
            "MRR@10 and lower/equal runtime.",
            "",
        ]

    @staticmethod
    def _append_section(lines: list[str], title: str, table: pd.DataFrame) -> None:
        lines.extend([f"## {title}", ""])
        if table.empty:
            lines.extend(["No data available.", ""])
            return

        preview = table.copy()
        preferred_columns = [
            column
            for column in BASE_COLUMNS + METRIC_COLUMNS + RUNTIME_COLUMNS
            + DERIVED_COLUMNS
            if column in preview.columns
        ]
        remaining_columns = [
            column for column in preview.columns if column not in preferred_columns
        ]
        preview = preview[preferred_columns + remaining_columns]
        lines.extend(
            [
                RecboleResultEvaluator._to_markdown_table(preview.head(20)),
                "",
            ]
        )

    @staticmethod
    def _to_markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "No data available."

        text_table = table.fillna("").astype(str)
        headers = list(text_table.columns)
        rows = text_table.values.tolist()

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _plot_best_metric(
        best_per_model: pd.DataFrame,
        metric: str,
        path: Path,
    ) -> None:
        labels = (
            best_per_model["experiment_type"].astype(str)
            + " | "
            + best_per_model["dataset"].astype(str)
            + " | "
            + best_per_model["model"].astype(str)
        )
        values = best_per_model[metric]
        height = max(4.0, 0.35 * len(best_per_model))

        plt.figure(figsize=(11, height))
        plt.barh(labels, values, color="#3572A5")
        plt.xlabel(metric)
        plt.title(f"Best {metric} per dataset and model")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _plot_runtime_vs_metric(
        best_per_model: pd.DataFrame,
        metric: str,
        path: Path,
    ) -> None:
        plt.figure(figsize=(9, 6))
        for model, group in best_per_model.groupby("model"):
            plt.scatter(
                group["runtime_seconds"],
                group[metric],
                label=str(model),
                s=70,
                alpha=0.85,
            )
        plt.xlabel("runtime_seconds")
        plt.ylabel(metric)
        plt.title(f"Runtime vs. best {metric}")
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _plot_runtime_distribution(results: pd.DataFrame, path: Path) -> None:
        labels = []
        values = []
        for model, group in results.groupby("model"):
            runtimes = group["runtime_seconds"].dropna()
            if not runtimes.empty:
                labels.append(str(model))
                values.append(runtimes)

        plt.figure(figsize=(10, 6))
        plt.boxplot(values, tick_labels=labels, vert=True)
        plt.ylabel("runtime_seconds")
        plt.title("Runtime distribution by model")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    @staticmethod
    def _experiment_type(name: str) -> str:
        lowered = name.lower()
        if "session" in lowered or "yoochoose" in lowered:
            return "session"
        if "topn" in lowered:
            return "topn"
        return lowered.replace("_full", "")

    @staticmethod
    def _clean_result_rows(results: pd.DataFrame) -> pd.DataFrame:
        cleaned = results.copy()

        if "dataset" in cleaned.columns:
            cleaned = cleaned[
                cleaned["dataset"].astype(str).str.lower() != "dataset"
            ]
        if "model" in cleaned.columns:
            cleaned = cleaned[
                cleaned["model"].astype(str).str.lower() != "model"
            ]

        subset = [
            column
            for column in ["experiment_name", "dataset", "model", "run_id"]
            if column in cleaned.columns
        ]
        if "run_id" in subset:
            cleaned = cleaned.drop_duplicates(subset=subset)
        else:
            cleaned = cleaned.drop_duplicates()

        return cleaned.reset_index(drop=True)

    @staticmethod
    def _pareto_front(quality: pd.Series, cost: pd.Series) -> pd.Series:
        pareto = pd.Series(False, index=quality.index)
        valid = quality.notna() & cost.notna()
        valid_quality = quality[valid]
        valid_cost = cost[valid]

        for index in valid_quality.index:
            dominates = (
                (valid_quality >= valid_quality.loc[index])
                & (valid_cost <= valid_cost.loc[index])
                & (
                    (valid_quality > valid_quality.loc[index])
                    | (valid_cost < valid_cost.loc[index])
                )
            )
            pareto.loc[index] = not bool(dominates.any())

        return pareto

    @staticmethod
    def _find_column(columns: Iterable[str], logical_name: str) -> str | None:
        for column in columns:
            if column.split(":", maxsplit=1)[0] == logical_name:
                return column
        return None

    @staticmethod
    def _format_timestamp(value: float) -> str | pd.NA:
        if pd.isna(value):
            return pd.NA
        return pd.to_datetime(value, unit="s", utc=True).isoformat()

    @staticmethod
    def _unique_configs(group: pd.DataFrame) -> int:
        if "config_json" not in group.columns:
            return len(group)
        return group["config_json"].fillna("{}").nunique()

    @staticmethod
    def _compact_values(values: list[object], max_values: int = 8) -> str:
        normalized = sorted({str(value) for value in values})
        if len(normalized) <= max_values:
            return ", ".join(normalized)
        return (
            ", ".join(normalized[:max_values])
            + f", ... ({len(normalized)} values)"
        )

    @staticmethod
    def _first_available_metric(results: pd.DataFrame) -> str:
        for metric in METRIC_COLUMNS:
            if metric in results.columns:
                return metric
        raise ValueError("No known metric columns found.")

    @staticmethod
    def parse_config_json(value: object) -> dict[str, object]:
        if pd.isna(value) or value == "":
            return {}
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create structured reports for RecBole tuning results.",
    )
    parser.add_argument(
        "--scope",
        choices=["full", "all"],
        default="full",
        help="Use only full tuning CSVs or every tuning result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for reports, tables, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = RecboleResultEvaluator(output_dir=args.output_dir)
    experiments = (
        evaluator.full_experiments()
        if args.scope == "full"
        else evaluator.all_experiments()
    )
    paths = evaluator.evaluate(experiments)

    print("Structured RecBole evaluation written to:")
    for name, path in sorted(paths.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
