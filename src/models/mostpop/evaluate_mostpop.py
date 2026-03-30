from pathlib import Path
import math
import pandas as pd


def load_data(file_path: Path, file_description: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description} file not found: {file_path}"
        )

    print(f"Loading {file_description}...")
    return pd.read_csv(file_path)


def build_ground_truth(test_df: pd.DataFrame) -> dict[int, int]:
    print("Building ground-truth dictionary from test data...")

    ground_truth = dict(zip(test_df["user_id"], test_df["item_id"]))
    return ground_truth


def build_recommendation_lists(recommendations_df: pd.DataFrame) -> dict[int, list[int]]:
    print("Building ranked recommendation lists per user...")

    recommendations_df = recommendations_df.sort_values(by=["user_id", "rank"])

    recommendation_lists = (
        recommendations_df.groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    return recommendation_lists


def hit_rate_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in recommended_items[:k] else 0.0


def ndcg_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    top_k_items = recommended_items[:k]

    if true_item not in top_k_items:
        return 0.0

    rank_index = top_k_items.index(true_item)
    return 1.0 / math.log2(rank_index + 2)

def mrr_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    top_k_items = recommended_items[:k]

    if true_item not in top_k_items:
        return 0.0

    rank_index = top_k_items.index(true_item)
    return 1.0 / (rank_index + 1)


def evaluate(
    ground_truth: dict[int, int],
    recommendation_lists: dict[int, list[int]]
) -> dict[str, float]:
    print("Evaluating MostPop recommendations...")

    hr_5_scores = []
    hr_10_scores = []
    ndcg_5_scores = []
    ndcg_10_scores = []
    mrr_5_scores = []
    mrr_10_scores = []

    common_users = sorted(set(ground_truth.keys()) & set(recommendation_lists.keys()))

    for user_id in common_users:
        true_item = ground_truth[user_id]
        recommended_items = recommendation_lists[user_id]

        hr_5_scores.append(hit_rate_at_k(recommended_items, true_item, k=5))
        hr_10_scores.append(hit_rate_at_k(recommended_items, true_item, k=10))
        ndcg_5_scores.append(ndcg_at_k(recommended_items, true_item, k=5))
        ndcg_10_scores.append(ndcg_at_k(recommended_items, true_item, k=10))
        mrr_5_scores.append(mrr_at_k(recommended_items, true_item, k=5))
        mrr_10_scores.append(mrr_at_k(recommended_items, true_item, k=10))

    results = {
        "HR@5": sum(hr_5_scores) / len(hr_5_scores),
        "HR@10": sum(hr_10_scores) / len(hr_10_scores),
        "NDCG@5": sum(ndcg_5_scores) / len(ndcg_5_scores),
        "NDCG@10": sum(ndcg_10_scores) / len(ndcg_10_scores),
        "evaluated_users": len(common_users),
        "MRR@5": sum(mrr_5_scores) / len(mrr_5_scores),
        "MRR@10": sum(mrr_10_scores) / len(mrr_10_scores),
    }

    return results


def save_results(results: dict[str, float], output_file: Path) -> None:
    print(f"Saving evaluation results to {output_file}...")

    df = pd.DataFrame([results])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_file = project_root / "results" / "movielens_mostpop_metrics.csv"

    test_file = project_root / "data" / "processed" / "movielens_test.csv"
    recommendations_file = project_root / "results" / "movielens_mostpop_recommendations.csv"

    test_df = load_data(test_file, "MovieLens test data")
    recommendations_df = load_data(recommendations_file, "MostPop recommendation output")

    print(f"\nTest interactions: {len(test_df):,}")
    print(f"Recommendation rows: {len(recommendations_df):,}")

    ground_truth = build_ground_truth(test_df)
    recommendation_lists = build_recommendation_lists(recommendations_df)

    results = evaluate(ground_truth, recommendation_lists)

    print("\nEvaluation results:")
    print(f"Evaluated users: {results['evaluated_users']:,}")
    print(f"HR@5: {results['HR@5']:.4f}")
    print(f"HR@10: {results['HR@10']:.4f}")
    print(f"NDCG@5: {results['NDCG@5']:.4f}")
    print(f"NDCG@10: {results['NDCG@10']:.4f}")
    print(f"MRR@5: {results['MRR@5']:.4f}")
    print(f"MRR@10: {results['MRR@10']:.4f}")

    save_results(results, output_file)

    print(f"Saved metrics file: {output_file}")


if __name__ == "__main__":
    main()