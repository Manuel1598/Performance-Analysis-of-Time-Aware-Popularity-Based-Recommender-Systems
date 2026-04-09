from pathlib import Path

from src.utils.io import load_data, save_recommendations, REQUIRED_INTERACTION_COLUMNS
from src.utils.recommendation import (
    build_user_seen_items,
    generate_model_recommendations_for_test_users,
)
from src.models.popularity.recentpop import RecentPopRecommender


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"
    output_file = project_root / "results" / "movielens_recentpop_recommendations.csv"

    train_df = load_data(
        train_file,
        "MovieLens training data",
        required_columns=REQUIRED_INTERACTION_COLUMNS
    )
    test_df = load_data(
        test_file,
        "MovieLens test data",
        required_columns=REQUIRED_INTERACTION_COLUMNS
    )

    print(f"\nTraining interactions: {len(train_df):,}")
    print(f"Training users: {train_df['user_id'].nunique():,}")
    print(f"Training items: {train_df['item_id'].nunique():,}")

    print(f"\nTest interactions: {len(test_df):,}")
    print(f"Test users: {test_df['user_id'].nunique():,}")

    user_seen = build_user_seen_items(train_df)

    model = RecentPopRecommender(window_days=30)
    model.fit(train_df)

    sample_row = test_df.iloc[0]
    sample_user_id = int(sample_row["user_id"])
    sample_timestamp = int(sample_row["timestamp"])

    print(f"\nSample user: {sample_user_id}")
    print(f"Reference timestamp (t0): {sample_timestamp}")

    sample_recommendations = model.recommend(
        user_id=sample_user_id,
        user_seen=user_seen,
        top_k=10,
        reference_timestamp=sample_timestamp,
    )

    print(f"\nRecentPop recommendations for user {sample_user_id}:")
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
        output_file=output_file
    )

    print("\nSaved file:")
    print(output_file)
    print(f"Rows: {len(recommendations_df):,}")


if __name__ == "__main__":
    main()