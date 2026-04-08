from pathlib import Path
import pandas as pd

from src.utils.io import load_data, save_recommendations, REQUIRED_INTERACTION_COLUMNS
from src.utils.recommendation import (
    build_user_seen_items,
    generate_recommendations_for_test_users,
)


def compute_item_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    print("Computing item popularity from training interactions...")

    popularity_df = (
        train_df.groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
        .sort_values(by="interaction_count", ascending=False)
        .reset_index(drop=True)
    )

    return popularity_df


def recommend_mostpop(
    user_id: int,
    popularity_df: pd.DataFrame,
    user_seen: dict[int, set[int]],
    top_k: int = 10
) -> list[int]:
    seen_items = user_seen.get(user_id, set())
    recommendations = []

    for item_id in popularity_df["item_id"]:
        if item_id not in seen_items:
            recommendations.append(item_id)

        if len(recommendations) == top_k:
            break

    return recommendations



def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"
    output_file = project_root / "results" / "movielens_mostpop_recommendations.csv"

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

    popularity_df = compute_item_popularity(train_df)
    user_seen = build_user_seen_items(train_df)

    print("\nTop 10 most popular items:")
    print(popularity_df.head(10))

    all_recommendations = generate_recommendations_for_test_users(
        test_df=test_df,
        recommend_fn=recommend_mostpop,
        use_reference_timestamp=False,
        popularity_df=popularity_df,
        user_seen=user_seen,
        top_k=10
    )

    print(f"\nGenerated recommendations for {len(all_recommendations):,} users.")

    sample_user_ids = list(all_recommendations.keys())[:3]
    for user_id in sample_user_ids:
        print(f"\nSample recommendations for user {user_id}:")
        print(all_recommendations[user_id])

    recommendations_df = save_recommendations(
        recommendations=all_recommendations,
        output_file=output_file
    )

    print("\nRecommendation output summary:")
    print(f"Saved file: {output_file}")
    print(f"Rows saved: {len(recommendations_df):,}")
    print("\nPreview:")
    print(recommendations_df.head(10))


if __name__ == "__main__":
    main()