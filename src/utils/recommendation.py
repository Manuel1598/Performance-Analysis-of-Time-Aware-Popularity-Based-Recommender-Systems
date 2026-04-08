import pandas as pd


def build_user_seen_items(train_df: pd.DataFrame) -> dict[int, set[int]]:
    print("Building user seen-item sets...")

    return (
        train_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )


def build_ground_truth(test_df: pd.DataFrame) -> dict[int, int]:
    print("Building ground-truth dictionary from test data...")

    return dict(zip(test_df["user_id"], test_df["item_id"]))


def build_recommendation_lists(
    recommendations_df: pd.DataFrame
) -> dict[int, list[int]]:
    print("Building ranked recommendation lists per user...")

    recommendations_df = recommendations_df.sort_values(by=["user_id", "rank"])

    return (
        recommendations_df.groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )


def generate_recommendations_for_test_users(
    test_df: pd.DataFrame,
    recommend_fn,
    use_reference_timestamp: bool = False,
    **recommend_kwargs
) -> dict[int, list[int]]:
    print(f"Generating recommendations for all test users (n={len(test_df):,})...")

    recommendations = {}

    if use_reference_timestamp:
        for _, row in test_df.iterrows():
            user_id = int(row["user_id"])
            reference_timestamp = int(row["timestamp"])

            recommendations[user_id] = recommend_fn(
                user_id=user_id,
                reference_timestamp=reference_timestamp,
                **recommend_kwargs
            )
    else:
        test_user_ids = test_df["user_id"].unique()

        for user_id in test_user_ids:
            recommendations[int(user_id)] = recommend_fn(
                user_id=int(user_id),
                **recommend_kwargs
            )

    return recommendations