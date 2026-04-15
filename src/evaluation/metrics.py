import math


def hit_rate_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in recommended_items[:k] else 0.0


def ndcg_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    top_k_items = recommended_items[:k]

    if true_item not in top_k_items:
        return 0.0

    rank_index = top_k_items.index(true_item)
    return 1.0 / math.log2(rank_index + 2)


def mrr_at_k(recommended_items: list[int], true_item: int, k: int) -> float:
    top_k_items = recommended_items[:k]

    if true_item not in top_k_items:
        return 0.0

    rank_index = top_k_items.index(true_item)
    return 1.0 / (rank_index + 1)

def coverage(
    recommendation_lists: dict[int, list[int]],
    total_item_count: int
) -> float:
    if total_item_count == 0:
        return 0.0

    recommended_items = set()

    for items in recommendation_lists.values():
        recommended_items.update(items)

    return len(recommended_items) / total_item_count

