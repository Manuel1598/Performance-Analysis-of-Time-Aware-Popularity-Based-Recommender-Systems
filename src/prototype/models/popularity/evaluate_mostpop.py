from pathlib import Path

from src.prototype.utils.io import load_data, save_results
from src.prototype.utils.recommendation import build_ground_truth, build_recommendation_lists
from src.prototype.evaluation.evaluator import evaluate_recommendations


DATASET_CONFIGS = {
    "movielens": {
        "label": "MovieLens",
        "train_file": "data/processed/movielens_train.csv",
        "test_file": "data/processed/movielens_test.csv",
        "recommendations_file": "results_prototype/movielens_mostpop_recommendations.csv",
        "output_file": "results_prototype/movielens_mostpop_metrics.csv",
    },
    "amazon": {
        "label": "Amazon",
        "train_file": "data/processed/amazon_train.csv",
        "test_file": "data/processed/amazon_test.csv",
        "recommendations_file": "results_prototype/amazon_mostpop_recommendations.csv",
        "output_file": "results_prototype/amazon_mostpop_metrics.csv",
    },
}


def evaluate_for_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )

    project_root = Path(__file__).resolve().parents[3]
    config = DATASET_CONFIGS[dataset_name]

    train_file = project_root / config["train_file"]
    test_file = project_root / config["test_file"]
    recommendations_file = project_root / config["recommendations_file"]
    output_file = project_root / config["output_file"]

    train_df = load_data(train_file, f"{config['label']} training data")
    test_df = load_data(test_file, f"{config['label']} test data")
    recommendations_df = load_data(
        recommendations_file,
        f"{config['label']} MostPop recommendation output"
    )

    print(f"\nDataset: {config['label']}")
    print(f"Training interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")
    print(f"Recommendation rows: {len(recommendations_df):,}")

    total_item_count = train_df["item_id"].nunique()

    ground_truth = build_ground_truth(test_df)
    recommendation_lists = build_recommendation_lists(recommendations_df)

    results = evaluate_recommendations(
        ground_truth=ground_truth,
        recommendation_lists=recommendation_lists,
        total_item_count=total_item_count,
        model_name="MostPop"
    )

    print("\nEvaluation results_prototype:")
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


def main() -> None:

    evaluate_for_dataset("movielens")
    evaluate_for_dataset("amazon")



if __name__ == "__main__":
    main()