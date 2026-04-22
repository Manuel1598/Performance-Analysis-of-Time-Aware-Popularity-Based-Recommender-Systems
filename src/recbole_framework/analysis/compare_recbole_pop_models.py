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

    mostpop_file = (
        project_root / "recbole_results"  / "movielens_mostpop_recbole_metrics.csv"
    )
    recentpop_file = (
        project_root / "recbole_results"  / "movielens_recentpop_recbole_metrics.csv"
    )
    decaypop_file = (
        project_root / "recbole_results"  / "movielens_decaypop_recbole_metrics.csv"
    )

    output_dir = project_root / "recbole_results" /  "analysis_results"
    output_table = output_dir / "movielens_recbole_pop_model_comparison.csv"
    output_plot_hit10 = output_dir / "movielens_recbole_hit10_comparison.png"
    output_plot_ndcg10 = output_dir / "movielens_recbole_ndcg10_comparison.png"
    output_plot_mrr10 = output_dir / "movielens_recbole_mrr10_comparison.png"

    mostpop_df = load_metrics(mostpop_file, "MostPop")
    recentpop_df = load_metrics(recentpop_file, "RecentPop")
    decaypop_df = load_metrics(decaypop_file, "DecayPop")

    comparison_df = pd.concat(
        [mostpop_df, recentpop_df, decaypop_df],
        ignore_index=True,
    )

    comparison_df = comparison_df[
        ["model", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10"]
    ]

    print("\nRecBole popularity model comparison:")
    print(comparison_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_table, index=False)
    print(f"\nSaved comparison table to: {output_table}")

    # Plot hit@10
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["hit@10"])
    plt.title("Hit@10 Comparison of RecBole Popularity Models")
    plt.xlabel("Model")
    plt.ylabel("Hit@10")
    plt.tight_layout()
    plt.savefig(output_plot_hit10)
    plt.close()
    print(f"Saved Hit@10 plot to: {output_plot_hit10}")

    # Plot ndcg@10
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["ndcg@10"])
    plt.title("NDCG@10 Comparison of RecBole Popularity Models")
    plt.xlabel("Model")
    plt.ylabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(output_plot_ndcg10)
    plt.close()
    print(f"Saved NDCG@10 plot to: {output_plot_ndcg10}")

    # Plot mrr@10
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["mrr@10"])
    plt.title("MRR@10 Comparison of RecBole Popularity Models")
    plt.xlabel("Model")
    plt.ylabel("MRR@10")
    plt.tight_layout()
    plt.savefig(output_plot_mrr10)
    plt.close()
    print(f"Saved MRR@10 plot to: {output_plot_mrr10}")


if __name__ == "__main__":
    main()