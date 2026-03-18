from pathlib import Path
import pandas as pd


def load_data(file_path: Path, file_description: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description} file not found: {file_path}"
        )

    print(f"Loading {file_description}...")
    df = pd.read_csv(file_path)

    required_columns = ["user_id", "item_id", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_description}: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


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


def build_user_seen_items(train_df: pd.DataFrame) -> dict[int, set[int]]:
    print("Building user seen-item sets...")

    user_seen = (
        train_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    return user_seen


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


def generate_recommendations_for_test_users(
    test_df: pd.DataFrame,
    popularity_df: pd.DataFrame,
    user_seen: dict[int, set[int]],
    top_k: int = 10
) -> dict[int, list[int]]:
    print(f"Generating MostPop recommendations for all test users (top-{top_k})...")

    test_user_ids = test_df["user_id"].unique()
    recommendations = {}

    for user_id in test_user_ids:
        recommendations[user_id] = recommend_mostpop(
            user_id=user_id,
            popularity_df=popularity_df,
            user_seen=user_seen,
            top_k=top_k
        )

    return recommendations


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"

    train_df = load_data(train_file, "MovieLens training data")
    test_df = load_data(test_file, "MovieLens test data")

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
        popularity_df=popularity_df,
        user_seen=user_seen,
        top_k=10
    )

    print(f"\nGenerated recommendations for {len(all_recommendations):,} users.")

    sample_user_ids = list(all_recommendations.keys())[:3]
    for user_id in sample_user_ids:
        print(f"\nSample recommendations for user {user_id}:")
        print(all_recommendations[user_id])


if __name__ == "__main__":
    main()