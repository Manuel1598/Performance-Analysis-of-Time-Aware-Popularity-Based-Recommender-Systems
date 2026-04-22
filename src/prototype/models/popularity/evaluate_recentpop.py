from pathlib import Path

from src.prototype.utils.io import load_data, save_results
from src.prototype.utils.recommendation import build_ground_truth, build_recommendation_lists
from src.prototype.evaluation.evaluator import evaluate_recommendations


DATASET_CONFIGS = {
    "movielens": {
        "label": "MovieLens",
        "train_file": "data/processed/movielens_train.csv",
        "test_file": "data/processed/movielens_test.csv",
        "recommendations_file": "results_prototype/movielens_recentpop_recommendations.csv",
        "output_file": "results_prototype/movielens_recentpop_metrics.csv",
    },
    "amazon": {
        "label": "Amazon",
        "train_file": "data/processed/amazon_train.csv",
        "test_file": "data/processed/amazon_test.csv",
        "recommendations_file": "results_prototype/amazon_recentpop_recommendations.csv",
        "output_file": "results_prototype/amazon_recentpop_metrics.csv",
    },
}


def evaluate_for_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    project_root = Path(__file__).resolve().parents[3]
    config = DATASET_CONFIGS[dataset_name]

    train_df = load_data(project_root / config["train_file"], "train")
    test_df = load_data(project_root / config["test_file"], "test")
    rec_df = load_data(project_root / config["recommendations_file"], "rec")

    print(f"\nDataset: {config['label']}")

    total_item_count = train_df["item_id"].nunique()

    ground_truth = build_ground_truth(test_df)
    recommendation_lists = build_recommendation_lists(rec_df)

    results = evaluate_recommendations(
        ground_truth=ground_truth,
        recommendation_lists=recommendation_lists,
        total_item_count=total_item_count,
        model_name="RecentPop",
    )

    print("\nEvaluation results_prototype:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    save_results(results, project_root / config["output_file"])


def main():
    evaluate_for_dataset("movielens")
    evaluate_for_dataset("amazon")


if __name__ == "__main__":
    main()