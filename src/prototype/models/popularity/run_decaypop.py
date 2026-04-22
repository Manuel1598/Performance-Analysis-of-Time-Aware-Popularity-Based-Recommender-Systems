from pathlib import Path

from src.prototype.utils.io import load_data, save_recommendations, REQUIRED_INTERACTION_COLUMNS
from src.prototype.utils.recommendation import (
    build_user_seen_items,
    generate_model_recommendations_for_test_users,
)
from src.prototype.models.popularity.decaypop import DecayPopRecommender


DATASET_CONFIGS = {
    "movielens": {
        "label": "MovieLens",
        "train_file": "data/processed/movielens_train.csv",
        "test_file": "data/processed/movielens_test.csv",
        "output_file": "results_prototype/movielens_decaypop_recommendations.csv",
    },
    "amazon": {
        "label": "Amazon",
        "train_file": "data/processed/amazon_train.csv",
        "test_file": "data/processed/amazon_test.csv",
        "output_file": "results_prototype/amazon_decaypop_recommendations.csv",
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
    print(f"Training users: {train_df['user_id'].nunique():,}")
    print(f"Training items: {train_df['item_id'].nunique():,}")

    print(f"\nTest interactions: {len(test_df):,}")
    print(f"Test users: {test_df['user_id'].nunique():,}")

    user_seen = build_user_seen_items(train_df)

    model = DecayPopRecommender(decay_lambda=1e-7)
    model.fit(train_df)

    sample_row = test_df.iloc[0]
    sample_user_id = sample_row["user_id"]
    sample_timestamp = int(sample_row["timestamp"])

    print(f"\nSample user: {sample_user_id}")
    print(f"Reference timestamp (t0): {sample_timestamp}")

    sample_recommendations = model.recommend(
        user_id=sample_user_id,
        user_seen=user_seen,
        top_k=10,
        reference_timestamp=sample_timestamp,
    )

    print(f"\nDecayPop recommendations for user {sample_user_id}:")
    print(sample_recommendations)

    all_recommendations = generate_model_recommendations_for_test_users(
        model=model,
        test_df=test_df,
        user_seen=user_seen,
        use_reference_timestamp=True,
        top_k=10,
    )

    print(f"\nGenerated recommendations for {len(all_recommendations):,} users.")

    recommendations_df = save_recommendations(
        recommendations=all_recommendations,
        output_file=project_root / config["output_file"],
    )

    print("\nRecommendation output summary:")
    print(f"Rows saved: {len(recommendations_df):,}")
    print("\nPreview:")
    print(recommendations_df.head(10))


def main() -> None:
    run_for_dataset("movielens")
    run_for_dataset("amazon")


if __name__ == "__main__":
    main()