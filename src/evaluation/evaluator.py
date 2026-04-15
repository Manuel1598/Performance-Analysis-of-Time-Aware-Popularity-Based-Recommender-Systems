from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k, coverage


def evaluate_recommendations(
    ground_truth: dict[int, int],
    recommendation_lists: dict[int, list[int]],
    total_item_count: int,
    model_name: str = "recommendation model"
) -> dict[str, float]:
    print(f"Evaluating {model_name} recommendations...")

    hr_5_scores = []
    hr_10_scores = []
    ndcg_5_scores = []
    ndcg_10_scores = []
    mrr_5_scores = []
    mrr_10_scores = []

    common_users = sorted(set(ground_truth.keys()) & set(recommendation_lists.keys()))

    coverage_score = coverage(
        recommendation_lists=recommendation_lists,
        total_item_count=total_item_count
    )

    for user_id in common_users:
        true_item = ground_truth[user_id]
        recommended_items = recommendation_lists[user_id]

        hr_5_scores.append(hit_rate_at_k(recommended_items, true_item, k=5))
        hr_10_scores.append(hit_rate_at_k(recommended_items, true_item, k=10))
        ndcg_5_scores.append(ndcg_at_k(recommended_items, true_item, k=5))
        ndcg_10_scores.append(ndcg_at_k(recommended_items, true_item, k=10))
        mrr_5_scores.append(mrr_at_k(recommended_items, true_item, k=5))
        mrr_10_scores.append(mrr_at_k(recommended_items, true_item, k=10))



    return {
        "HR@5": sum(hr_5_scores) / len(hr_5_scores),
        "HR@10": sum(hr_10_scores) / len(hr_10_scores),
        "NDCG@5": sum(ndcg_5_scores) / len(ndcg_5_scores),
        "NDCG@10": sum(ndcg_10_scores) / len(ndcg_10_scores),
        "MRR@5": sum(mrr_5_scores) / len(mrr_5_scores),
        "MRR@10": sum(mrr_10_scores) / len(mrr_10_scores),
        "Coverage": coverage_score,
        "evaluated_users": len(common_users),
    }