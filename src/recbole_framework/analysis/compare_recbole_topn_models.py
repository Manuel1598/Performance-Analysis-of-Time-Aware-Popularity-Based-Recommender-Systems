from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ["movielens", "amazon"]

MODELS = {
    "mostpop": "MostPop",
    "recentpop": "RecentPop",
    "decaypop": "DecayPop",
    "bpr": "BPR",
}


def load_metrics(file_path: Path, dataset_name: str, model_name: str) -> pd.DataFrame | None:
    if not file_path.exists():
        print(f"Skipping missing metrics file: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["dataset"] = dataset_name
    df["model"] = model_name
    return df


def plot_metric(comparison_df: pd.DataFrame, metric: str, output_file: Path) -> None:
    pivot_df = comparison_df.pivot(index="model", columns="dataset", values=metric)

    ax = pivot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"{metric.upper()} Comparison of RecBole Models")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved {metric} plot to: {output_file}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    results_dir = project_root / "recbole_results"
    output_dir = results_dir / "analysis_results"

    loaded_results = []

    for dataset_key in DATASETS:
        for model_key, model_name in MODELS.items():
            metrics_file = results_dir / f"{dataset_key}_{model_key}_recbole_metrics.csv"

            metrics_df = load_metrics(
                file_path=metrics_file,
                dataset_name=dataset_key,
                model_name=model_name,
            )

            if metrics_df is not None:
                loaded_results.append(metrics_df)

    if not loaded_results:
        raise ValueError("No RecBole metric files found.")

    comparison_df = pd.concat(loaded_results, ignore_index=True)

    comparison_df = comparison_df[
        ["dataset", "model", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]
    ]

    comparison_df = comparison_df.sort_values(["dataset", "model"])

    print("\nRecBole model comparison:")
    print(comparison_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_table = output_dir / "recbole_model_comparison.csv"
    comparison_df.to_csv(output_table, index=False)
    print(f"\nSaved comparison table to: {output_table}")

    plot_metric(
        comparison_df=comparison_df,
        metric="hit@10",
        output_file=output_dir / "recbole_hit10_comparison.png",
    )

    plot_metric(
        comparison_df=comparison_df,
        metric="ndcg@10",
        output_file=output_dir / "recbole_ndcg10_comparison.png",
    )

    plot_metric(
        comparison_df=comparison_df,
        metric="mrr@10",
        output_file=output_dir / "recbole_mrr10_comparison.png",
    )


if __name__ == "__main__":
    main()