from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "mostpop": {
        "display_name": "MostPop",
        "file_name": "yoochoose_sample_mostpop_recbole_metrics.csv",
    },
    "vsknn": {
        "display_name": "VS-KNN",
        "file_name": "yoochoose_sample_vsknn_sequential_recbole_metrics.csv",
    },
    "vstan": {
        "display_name": "VSTAN",
        "file_name": "yoochoose_sample_vstan_recbole_metrics.csv",
    },
}


def load_metrics(file_path: Path, model_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["model"] = model_name
    return df


def plot_metric(comparison_df: pd.DataFrame, metric: str, output_file: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df[metric])
    plt.title(f"{metric.upper()} Comparison of Session-based RecBole Models")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Saved {metric} plot to: {output_file}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    results_dir = project_root / "recbole_results"
    output_dir = results_dir / "analysis_results"

    loaded_results = []

    for model_config in MODELS.values():
        metrics_file = results_dir / model_config["file_name"]

        metrics_df = load_metrics(
            file_path=metrics_file,
            model_name=model_config["display_name"],
        )

        loaded_results.append(metrics_df)

    comparison_df = pd.concat(loaded_results, ignore_index=True)

    comparison_df = comparison_df[
        ["model", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]
    ]

    print("\nSession-based RecBole model comparison:")
    print(comparison_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_table = output_dir / "yoochoose_sample_session_model_comparison.csv"
    comparison_df.to_csv(output_table, index=False)

    print(f"\nSaved comparison table to: {output_table}")

    plot_metric(
        comparison_df=comparison_df,
        metric="hit@10",
        output_file=output_dir / "yoochoose_sample_session_hit10_comparison.png",
    )

    plot_metric(
        comparison_df=comparison_df,
        metric="ndcg@10",
        output_file=output_dir / "yoochoose_sample_session_ndcg10_comparison.png",
    )

    plot_metric(
        comparison_df=comparison_df,
        metric="mrr@10",
        output_file=output_dir / "yoochoose_sample_session_mrr10_comparison.png",
    )


if __name__ == "__main__":
    main()