from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(file_path: Path, model_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["model"] = model_name
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    mostpop_file = project_root / "results" / "movielens_mostpop_metrics.csv"
    recentpop_file = project_root / "results" / "movielens_recentpop_metrics.csv"
    decaypop_file = project_root / "results" / "movielens_decaypop_metrics.csv"

    output_table = project_root / "results" / "analysis_results" / "movielens_popularity_model_comparison.csv"
    output_plot_hr10 = project_root / "results" / "analysis_results" / "movielens_hr10_comparison.png"
    output_plot_ndcg10 = project_root / "results" / "analysis_results" / "movielens_ndcg10_comparison.png"

    mostpop_df = load_metrics(mostpop_file, "MostPop")
    recentpop_df = load_metrics(recentpop_file, "RecentPop")
    decaypop_df = load_metrics(decaypop_file, "DecayPop")

    comparison_df = pd.concat(
        [mostpop_df, recentpop_df, decaypop_df],
        ignore_index=True
    )

    comparison_df = comparison_df[
        ["model", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "evaluated_users"]
    ]

    print("\nComparison table:")
    print(comparison_df)

    output_table.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_table, index=False)
    print(f"\nSaved comparison table to: {output_table}")

    # Plot HR@10
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["HR@10"])
    plt.title("HR@10 Comparison of Popularity-Based Models")
    plt.xlabel("Model")
    plt.ylabel("HR@10")
    plt.tight_layout()
    plt.savefig(output_plot_hr10)
    plt.close()
    print(f"Saved HR@10 plot to: {output_plot_hr10}")

    # Plot NDCG@10
    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["NDCG@10"])
    plt.title("NDCG@10 Comparison of Popularity-Based Models")
    plt.xlabel("Model")
    plt.ylabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(output_plot_ndcg10)
    plt.close()
    print(f"Saved NDCG@10 plot to: {output_plot_ndcg10}")


if __name__ == "__main__":
    main()