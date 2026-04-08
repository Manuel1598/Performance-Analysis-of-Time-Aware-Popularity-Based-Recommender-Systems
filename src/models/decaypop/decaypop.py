from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import load_data, save_recommendations, REQUIRED_INTERACTION_COLUMNS
from src.utils.recommendation import (
    build_user_seen_items,
    generate_recommendations_for_test_users,
)




def compute_decay_popularity(
    train_df: pd.DataFrame,
    reference_timestamp: int,
    decay_lambda: float = 1e-7
) -> pd.DataFrame:
    print(f"Computing DecayPop popularity for reference time {reference_timestamp}...")

    df = train_df.copy()

    # Nur Interaktionen berücksichtigen, die vor oder genau bei t0 liegen
    df = df[df["timestamp"] <= reference_timestamp].copy()

    # Zeitdifferenz in Sekunden
    df["time_diff"] = reference_timestamp - df["timestamp"]

    # Exponentielles Decay-Gewicht
    df["weight"] = np.exp(-decay_lambda * df["time_diff"])

    popularity_df = (
        df.groupby("item_id")["weight"]
        .sum()
        .reset_index()
        .sort_values(by="weight", ascending=False)
        .reset_index(drop=True)
    )

    return popularity_df


def recommend_decaypop(
    user_id: int,
    reference_timestamp: int,
    train_df: pd.DataFrame,
    user_seen: dict[int, set[int]],
    top_k: int = 10,
    decay_lambda: float = 1e-7
) -> list[int]:
    seen_items = user_seen.get(user_id, set())

    popularity_df = compute_decay_popularity(
        train_df=train_df,
        reference_timestamp=reference_timestamp,
        decay_lambda=decay_lambda
    )

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
    output_file = project_root / "results" / "movielens_decaypop_recommendations.csv"

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

    # Beispiel-User zum Testen
    sample_row = test_df.iloc[0]
    sample_user_id = int(sample_row["user_id"])
    sample_timestamp = int(sample_row["timestamp"])

    print(f"\nSample user: {sample_user_id}")
    print(f"Reference timestamp (t0): {sample_timestamp}")

    recommendations = recommend_decaypop(
        user_id=sample_user_id,
        reference_timestamp=sample_timestamp,
        train_df=train_df,
        user_seen=user_seen,
        top_k=10,
        decay_lambda=1e-7
    )

    print(f"\nDecayPop recommendations for user {sample_user_id}:")
    print(recommendations)

    all_recommendations = generate_recommendations_for_test_users(
        test_df=test_df,
        recommend_fn=recommend_decaypop,
        use_reference_timestamp=True,
        train_df=train_df,
        user_seen=user_seen,
        top_k=10,
        decay_lambda=1e-7
    )



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