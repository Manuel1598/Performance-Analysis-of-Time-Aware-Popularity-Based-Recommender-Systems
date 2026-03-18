from pathlib import Path
import pandas as pd


def load_train_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Train file not found: {file_path}\n"
            "Please run split.py first."
        )

    print("Loading MovieLens training data...")
    df = pd.read_csv(file_path)

    required_columns = ["user_id", "item_id", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"

    train_df = load_train_data(train_file)

    print(f"Training interactions: {len(train_df):,}")
    print(f"Training users: {train_df['user_id'].nunique():,}")
    print(f"Training items: {train_df['item_id'].nunique():,}")

    popularity_df = compute_item_popularity(train_df)
    user_seen = build_user_seen_items(train_df)

    print("\nTop 10 most popular items:")
    print(popularity_df.head(10))

    sample_user_id = train_df["user_id"].iloc[0]
    recommendations = recommend_mostpop(
        user_id=sample_user_id,
        popularity_df=popularity_df,
        user_seen=user_seen,
        top_k=10
    )

    print(f"\nSample recommendations for user {sample_user_id}:")
    print(recommendations)


if __name__ == "__main__":
    main()