from pathlib import Path

from src.utils.io import load_data, save_recommendations, REQUIRED_INTERACTION_COLUMNS
from src.utils.recommendation import (
    build_user_seen_items,
    generate_model_recommendations_for_test_users,
)
from src.models.popularity.recentpop import RecentPopRecommender


DATASET_CONFIGS = {
    "movielens": {
        "label": "MovieLens",
        "train_file": "data/processed/movielens_train.csv",
        "test_file": "data/processed/movielens_test.csv",
        "output_file": "results/movielens_recentpop_recommendations.csv",
    },
    "amazon": {
        "label": "Amazon",
        "train_file": "data/processed/amazon_train.csv",
        "test_file": "data/processed/amazon_test.csv",
        "output_file": "results/amazon_recentpop_recommendations.csv",
    },
}


def run_for_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    project_root = Path(__file__).resolve().parents[3]
    config = DATASET_CONFIGS[dataset_name]

    train_df = load_data(
        project_root / config["train_file"],
        f"{config['label']} training data",
        required_columns=REQUIRED_INTERACTION_COLUMNS,
    )

    test_df = load_data(
        project_root / config["test_file"],
        f"{config['label']} test data",
        required_columns=REQUIRED_INTERACTION_COLUMNS,
    )

    print(f"\nDataset: {config['label']}")
    print(f"Training interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")

    user_seen = build_user_seen_items(train_df)

    model = RecentPopRecommender(window_days=30)
    model.fit(train_df)

    # Sample check
    sample_row = test_df.iloc[0]
    sample_user = sample_row["user_id"]
    sample_ts = sample_row["timestamp"]

    print(f"\nSample user: {sample_user}")
    print(f"Reference timestamp: {sample_ts}")

    sample_rec = model.recommend(
        user_id=sample_user,
        user_seen=user_seen,
        top_k=10,
        reference_timestamp=sample_ts,
    )

    print("\nSample recommendations:")
    print(sample_rec)

    all_recommendations = generate_model_recommendations_for_test_users(
        model=model,
        test_df=test_df,
        user_seen=user_seen,
        use_reference_timestamp=True,
        top_k=10,
    )

    recommendations_df = save_recommendations(
        recommendations=all_recommendations,
        output_file=project_root / config["output_file"],
    )

    print("\nSaved recommendations:")
    print(recommendations_df.head())


def main():
    run_for_dataset("movielens")
    run_for_dataset("amazon")


if __name__ == "__main__":
    main()