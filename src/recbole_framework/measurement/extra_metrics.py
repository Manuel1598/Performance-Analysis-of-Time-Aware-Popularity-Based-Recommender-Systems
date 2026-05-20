import pandas as pd


def calculate_coverage(recommendations: list[list[int]], total_items: int) -> float:
    recommended_items = set()

    for rec_list in recommendations:
        recommended_items.update(rec_list)

    if total_items == 0:
        return 0.0

    return len(recommended_items) / total_items


def calculate_average_recommendation_popularity(
    recommendations: list[list[int]],
    train_interactions: pd.DataFrame,
    item_column: str = "item_id",
) -> float:
    item_popularity = train_interactions[item_column].value_counts().to_dict()

    popularity_values = []

    for rec_list in recommendations:
        for item_id in rec_list:
            popularity_values.append(item_popularity.get(item_id, 0))

    if not popularity_values:
        return 0.0

    return sum(popularity_values) / len(popularity_values)