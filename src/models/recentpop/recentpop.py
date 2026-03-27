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


def build_user_seen_items(train_df: pd.DataFrame) -> dict[int, set[int]]:
    print("Building user seen-item sets...")

    user_seen = (
        train_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    return user_seen


def compute_recent_popularity(
    train_df: pd.DataFrame,
    reference_timestamp: int,
    window_days: int = 30
) -> pd.DataFrame:
    print(f"Computing RecentPop popularity for reference time {reference_timestamp}...")

    window_seconds = window_days * 24 * 60 * 60
    window_start = reference_timestamp - window_seconds

    recent_df = train_df[
        (train_df["timestamp"] >= window_start) &
        (train_df["timestamp"] <= reference_timestamp)
    ].copy()

    popularity_df = (
        recent_df.groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
        .sort_values(by="interaction_count", ascending=False)
        .reset_index(drop=True)
    )

    return popularity_df


def recommend_recentpop(
    user_id: int,
    reference_timestamp: int,
    train_df: pd.DataFrame,
    user_seen: dict[int, set[int]],
    top_k: int = 10,
    window_days: int = 30
) -> list[int]:
    seen_items = user_seen.get(user_id, set())

    popularity_df = compute_recent_popularity(
        train_df=train_df,
        reference_timestamp=reference_timestamp,
        window_days=window_days
    )

    recommendations = []

    for item_id in popularity_df["item_id"]:
        if item_id not in seen_items:
            recommendations.append(item_id)

        if len(recommendations) == top_k:
            break

    return recommendations



def generate_recommendations_for_test_users(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user_seen: dict[int, set[int]],
    top_k: int = 10,
    window_days: int = 30
) -> dict[int, list[int]]:

    print(f"Generating RecentPop recommendations for all test users (top-{top_k})...")

    recommendations = {}

    for _, row in test_df.iterrows():
        user_id = int(row["user_id"])
        t0 = int(row["timestamp"])

        recs = recommend_recentpop(
            user_id=user_id,
            reference_timestamp=t0,
            train_df=train_df,
            user_seen=user_seen,
            top_k=top_k,
            window_days=window_days
        )

        recommendations[user_id] = recs

    return recommendations

def save_recommendations(
    recommendations: dict[int, list[int]],
    output_file: Path
) -> pd.DataFrame:

    print(f"Saving recommendations to {output_file}...")

    rows = []

    for user_id, items in recommendations.items():
        for rank, item_id in enumerate(items, start=1):
            rows.append({
                "user_id": user_id,
                "rank": rank,
                "item_id": item_id
            })

    df = pd.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df



def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"
    output_file = project_root / "results" / "movielens_recentpop_recommendations.csv"

    train_df = load_data(train_file, "MovieLens training data")
    test_df = load_data(test_file, "MovieLens test data")

    print(f"\nTraining interactions: {len(train_df):,}")
    print(f"Training users: {train_df['user_id'].nunique():,}")
    print(f"Training items: {train_df['item_id'].nunique():,}")

    print(f"\nTest interactions: {len(test_df):,}")
    print(f"Test users: {test_df['user_id'].nunique():,}")

    user_seen = build_user_seen_items(train_df)

    sample_row = test_df.iloc[0]
    sample_user_id = int(sample_row["user_id"])
    sample_timestamp = int(sample_row["timestamp"])

    print(f"\nSample user: {sample_user_id}")
    print(f"Reference timestamp (t0): {sample_timestamp}")

    recommendations = recommend_recentpop(
        user_id=sample_user_id,
        reference_timestamp=sample_timestamp,
        train_df=train_df,
        user_seen=user_seen,
        top_k=10,
        window_days=30
    )

    print(f"\nRecentPop recommendations for user {sample_user_id}:")
    print(recommendations)

    all_recommendations = generate_recommendations_for_test_users(
        test_df=test_df,
        train_df=train_df,
        user_seen=user_seen,
        top_k=10,
        window_days=30
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