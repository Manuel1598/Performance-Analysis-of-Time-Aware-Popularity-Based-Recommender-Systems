from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(file_path: Path, model_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["model"] = model_name
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    # =========================
    # Input files
    # =========================
    mostpop_file = project_root / "recbole_results" / "movielens_mostpop_recbole_metrics.csv"
    recentpop_file = project_root / "recbole_results" / "movielens_recentpop_recbole_metrics.csv"
    decaypop_file = project_root / "recbole_results" / "movielens_decaypop_recbole_metrics.csv"
    bpr_file = project_root / "recbole_results" / "movielens_bpr_recbole_metrics.csv"

    # =========================
    # Output paths
    # =========================
    output_dir = project_root / "recbole_results" / "analysis_results"

    output_table = output_dir / "movielens_recbole_model_comparison.csv"
    output_plot_hit10 = output_dir / "movielens_recbole_hit10_comparison.png"
    output_plot_ndcg10 = output_dir / "movielens_recbole_ndcg10_comparison.png"
    output_plot_mrr10 = output_dir / "movielens_recbole_mrr10_comparison.png"

    # =========================
    # Load data
    # =========================
    mostpop_df = load_metrics(mostpop_file, "MostPop")
    recentpop_df = load_metrics(recentpop_file, "RecentPop")
    decaypop_df = load_metrics(decaypop_file, "DecayPop")
    bpr_df = load_metrics(bpr_file, "BPR")

    # =========================
    # Combine
    # =========================
    comparison_df = pd.concat(
        [mostpop_df, recentpop_df, decaypop_df, bpr_df],
        ignore_index=True,
    )

    comparison_df = comparison_df[
        ["model", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]
    ]

    print("\nRecBole model comparison:")
    print(comparison_df)

    # =========================
    # Save table
    # =========================
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_table, index=False)

    print(f"\nSaved comparison table to: {output_table}")

    # =========================
    # Plot helper
    # =========================
    def plot_metric(metric: str, output_file: Path, title: str):
        plt.figure(figsize=(8, 5))
        plt.bar(comparison_df["model"], comparison_df[metric])
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Saved {metric} plot to: {output_file}")

    # =========================
    # Plots
    # =========================
    plot_metric("hit@10", output_plot_hit10, "Hit@10 Comparison of RecBole Models")
    plot_metric("ndcg@10", output_plot_ndcg10, "NDCG@10 Comparison of RecBole Models")
    plot_metric("mrr@10", output_plot_mrr10, "MRR@10 Comparison of RecBole Models")


if __name__ == "__main__":
    main()