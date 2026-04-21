from pathlib import Path

from src.prototype.utils.io import load_data, save_results
from src.prototype.utils.recommendation import build_ground_truth, build_recommendation_lists
from src.prototype.evaluation.evaluator import evaluate_recommendations


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"
    recommendations_file = project_root / "results" / "movielens_bpr_recommendations.csv"
    output_file = project_root / "results" / "movielens_bpr_metrics.csv"

    train_df = load_data(train_file, "MovieLens training data")
    test_df = load_data(test_file, "MovieLens test data")
    recommendations_df = load_data(recommendations_file, "BPR recommendation output")

    print(f"\nTraining interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")
    print(f"Recommendation rows: {len(recommendations_df):,}")

    total_item_count = train_df["item_id"].nunique()

    ground_truth = build_ground_truth(test_df)
    recommendation_lists = build_recommendation_lists(recommendations_df)

    results = evaluate_recommendations(
        ground_truth=ground_truth,
        recommendation_lists=recommendation_lists,
        total_item_count=total_item_count,
        model_name="BPR",
    )

    print("\nEvaluation results:")
    print(f"Evaluated users: {results['evaluated_users']:,}")
    print(f"HR@5: {results['HR@5']:.4f}")
    print(f"HR@10: {results['HR@10']:.4f}")
    print(f"NDCG@5: {results['NDCG@5']:.4f}")
    print(f"NDCG@10: {results['NDCG@10']:.4f}")
    print(f"MRR@5: {results['MRR@5']:.4f}")
    print(f"MRR@10: {results['MRR@10']:.4f}")
    print(f"Coverage: {results['Coverage']:.4f}")

    save_results(results, output_file)

    print(f"\nSaved metrics file: {output_file}")


if __name__ == "__main__":
    main()