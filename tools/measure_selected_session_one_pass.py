"""One-pass evaluation for expensive selected session configurations."""

from __future__ import annotations

import argparse
import json
import time

import torch
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from tools.measure_selected_session_beyond_accuracy import (
    MODEL_CLASSES,
    SELECTED_CONFIGS,
    build_config,
    save_result,
)


def measure(model, test_data, train_data, config, top_k: int = 10) -> dict:
    model.eval()
    item_field = config["ITEM_ID_FIELD"]
    n_items = train_data.dataset.num(item_field)
    catalogue_size = n_items - 1
    train_ids = train_data.dataset.inter_feat[item_field].long().cpu()
    popularity = torch.bincount(train_ids, minlength=n_items).float()

    unique_items: set[int] = set()
    popularity_sum = 0.0
    recommendation_count = 0
    target_count = 0
    totals = {name: 0.0 for name in ("hit5", "hit10", "ndcg5", "ndcg10", "mrr5", "mrr10")}
    started = time.perf_counter()

    with torch.no_grad():
        for interaction, history_index, positive_users, positive_items in test_data:
            interaction = interaction.to(config["device"])
            scores = model.full_sort_predict(interaction).view(-1, n_items)
            scores[:, 0] = -float("inf")
            if history_index is not None:
                scores[history_index] = -float("inf")
            top_items = torch.topk(scores, k=top_k, dim=1).indices.cpu()

            flat = top_items.reshape(-1)
            unique_items.update(int(item) for item in flat.tolist())
            popularity_sum += float(popularity[flat].sum().item())
            recommendation_count += int(flat.numel())

            users = positive_users.long().cpu()
            targets = positive_items.long().cpu()
            matches = top_items[users].eq(targets.unsqueeze(1))
            hit = matches.any(dim=1)
            ranks = matches.float().argmax(dim=1).long() + 1
            ranks = torch.where(hit, ranks, torch.zeros_like(ranks))
            target_count += int(ranks.numel())

            reciprocal = torch.where(ranks > 0, 1.0 / ranks.float(), 0.0)
            discount = torch.where(
                ranks > 0, 1.0 / torch.log2(ranks.float() + 1.0), 0.0
            )
            totals["hit10"] += float((ranks > 0).sum())
            totals["hit5"] += float(((ranks > 0) & (ranks <= 5)).sum())
            totals["mrr10"] += float(reciprocal.sum())
            totals["mrr5"] += float(torch.where(ranks <= 5, reciprocal, 0.0).sum())
            totals["ndcg10"] += float(discount.sum())
            totals["ndcg5"] += float(torch.where(ranks <= 5, discount, 0.0).sum())

    if target_count == 0:
        raise RuntimeError("No held-out targets were evaluated")
    return {
        "hit@5": totals["hit5"] / target_count,
        "hit@10": totals["hit10"] / target_count,
        "ndcg@5": totals["ndcg5"] / target_count,
        "ndcg@10": totals["ndcg10"] / target_count,
        "mrr@5": totals["mrr5"] / target_count,
        "mrr@10": totals["mrr10"] / target_count,
        "coverage@10": len(unique_items) / catalogue_size,
        "avg_recommendation_popularity@10": popularity_sum / recommendation_count,
        "unique_recommended_items@10": len(unique_items),
        "catalogue_size": catalogue_size,
        "recommendation_count": recommendation_count,
        "extra_metrics_runtime_seconds": round(time.perf_counter() - started, 2),
    }


def run(dataset_name: str, model_name: str, device: str) -> dict:
    model_class = MODEL_CLASSES[model_name]
    updates = SELECTED_CONFIGS[dataset_name][model_name]
    config = build_config(model_class, dataset_name, updates, device)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    started = time.perf_counter()
    trainer.fit(train_data, valid_data, saved=False, verbose=False)
    metrics = measure(model, test_data, train_data, config)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "implementation": "audited" if model_name == "VS-KNN" else "current",
        "device": device,
        **metrics,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "config_json": json.dumps(updates, sort_keys=True),
        "status": "success",
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SELECTED_CONFIGS), required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for dataset_name in args.datasets:
        for model_name in args.models:
            print(f"run {dataset_name} / {model_name}", flush=True)
            result = run(dataset_name, model_name, args.device)
            save_result(result)
            print(
                f"mrr@10={result['mrr@10']:.4f} "
                f"coverage@10={result['coverage@10']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
