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