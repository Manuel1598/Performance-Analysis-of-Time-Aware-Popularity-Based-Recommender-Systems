"""Inference-only, descriptive test stratification by available prefix length.

Uses trusted project checkpoints and frozen configurations. Never calls fit.
Publishes group metrics only when all six aggregate ranking metrics reproduce
the original rounded test result. Saves compact query outcomes for auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from tools.run_validation_first_experiments import SESSION_MODELS, build_config
from tools.consolidate_final_results import ROOT, OUTPUT, sha256

GROUPS = ("1", "2", "3", "4+")
METRICS = tuple(f"{m}@{k}" for k in (5, 10) for m in ("hit", "mrr", "ndcg"))


def init_worker(model, n_items):
    global WORKER_MODEL, WORKER_ITEMS
    torch.set_num_threads(1)
    WORKER_MODEL, WORKER_ITEMS = model, n_items


@torch.no_grad()
def score_batch(batch, model, n_items):
    interaction, history_index, positive_u, positive_i = batch
    interaction = interaction.to("cpu")
    n = len(interaction)
    if len(positive_u) != n or not torch.equal(positive_u.cpu(), torch.arange(n)):
        raise ValueError("Prefix stratification requires exactly one target per query")
    lengths = interaction[model.ITEM_SEQ_LEN].cpu().numpy()
    sequences = interaction[model.ITEM_SEQ].cpu().numpy()
    targets = positive_i.cpu().numpy()
    scores = model.full_sort_predict(interaction).view(n, n_items)
    if torch.isnan(scores).any():
        raise ValueError("NaN prediction score")
    scores[:, 0] = -float("inf")
    if history_index is not None:
        scores[history_index] = -float("inf")
    top = torch.topk(scores, 10, dim=-1).indices
    matches = top.eq(positive_i.to(top.device).view(-1, 1))
    ranks = torch.where(matches.any(dim=1), matches.int().argmax(dim=1) + 1, 0).cpu().numpy()
    return lengths, sequences, targets, ranks


def worker_score(batch):
    return score_batch(batch, WORKER_MODEL, WORKER_ITEMS)


def prefix_group(length: int) -> str:
    if length < 1:
        raise ValueError("A test prefix must contain at least one item")
    return str(length) if length <= 3 else "4+"


def metrics_from_ranks(ranks: np.ndarray, cutoff: int) -> dict[str, float]:
    if not len(ranks):
        return {f"{m}@{cutoff}": float("nan") for m in ("hit", "mrr", "ndcg")}
    hit = (ranks > 0) & (ranks <= cutoff)
    safe = np.maximum(ranks, 1)
    return {
        f"hit@{cutoff}": float(hit.mean()),
        f"mrr@{cutoff}": float(np.where(hit, 1.0 / safe, 0).mean()),
        f"ndcg@{cutoff}": float(np.where(hit, 1.0 / np.log2(safe + 1), 0).mean()),
    }


def reconcile(values: dict, original: dict, query_count: int) -> None:
    if query_count * 10 != int(original["recommendation_count"]):
        raise ValueError("Test query count differs from original final evaluation")
    for metric in METRICS:
        if abs(values[metric] - float(original[metric])) > 0.000051:
            raise ValueError(f"Aggregate mismatch {metric}: replay={values[metric]:.8f}, original={original[metric]}")


def replay(row: dict, destination: Path, workers: int = 1) -> None:
    dataset_name, model_name = row["dataset"], row["model"]
    stem = f"{dataset_name}__{model_name}"
    audit_path = destination / f"{stem}.json"
    checkpoint_path = ROOT / row["checkpoint_path"]
    input_path = ROOT / "data" / "recbole" / dataset_name / f"{dataset_name}.inter"
    fingerprint = {"checkpoint_sha256": sha256(checkpoint_path), "data_sha256": sha256(input_path),
                   "final_test_id": row["final_test_id"], "config_json": row["config_json"],
                   "evaluator_sha256": sha256(Path(__file__))}
    if audit_path.exists():
        prior = json.loads(audit_path.read_text())
        if prior.get("status") == "success" and prior.get("fingerprint") == fingerprint:
            print(f"Skipping verified replay: {stem}", flush=True)
            return
    started = time.perf_counter()
    print(f"Replaying {stem} (no training)", flush=True)
    # These are the user's own transferred/project-generated checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["dataset"] != dataset_name or int(config["seed"]) != 42:
        raise ValueError("Checkpoint identity differs from final row")
    updates = json.loads(row["config_json"])
    expected_config = build_config("session", SESSION_MODELS[model_name], dataset_name, updates, "cpu")
    for key in updates:
        if config[key] != expected_config[key]:
            raise ValueError(f"Checkpoint configuration mismatch: {key}")
    config["device"] = torch.device("cpu")
    config["use_gpu"] = False
    config["data_path"] = str(input_path.parent)
    config["show_progress"] = False
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = SESSION_MODELS[model_name](config, train_data.dataset).to("cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()
    queries, split_hash = [], hashlib.sha256()
    n_items = train_data.dataset.item_num
    worker_count = workers if model_name in ("VS-KNN", "VSTAN") else 1
    pool_context = ProcessPoolExecutor(max_workers=worker_count, initializer=init_worker, initargs=(model, n_items)) if worker_count > 1 else nullcontext(None)
    with pool_context as pool:
        results = pool.map(worker_score, test_data, chunksize=1) if pool else (score_batch(batch, model, n_items) for batch in test_data)
        for batch_index, (lengths, sequences, targets, rank) in enumerate(results):
            for array in (lengths, sequences, targets):
                split_hash.update(np.asarray(array, dtype="<i8").tobytes())
            queries.extend(zip(lengths.tolist(), rank.tolist()))
            if batch_index % 20 == 0:
                print(f"  {stem}: {len(queries)} queries", flush=True)
    q = pd.DataFrame(queries, columns=["prefix_length", "target_rank_at_10"])
    overall = {}
    for cutoff in (5, 10):
        overall.update(metrics_from_ranks(q.target_rank_at_10.to_numpy(), cutoff))
    reconcile(overall, row, len(q))
    q["group"] = q.prefix_length.map(prefix_group)
    groups = []
    for group in GROUPS:
        subset = q[q.group == group]
        record = {"dataset": dataset_name, "model": model_name, "prefix_group": group,
                  "query_count": len(subset), "query_share": len(subset) / len(q),
                  "final_test_id": row["final_test_id"]}
        for cutoff in (5, 10):
            record.update(metrics_from_ranks(subset.target_rank_at_10.to_numpy(), cutoff))
        groups.append(record)
    grouped = pd.DataFrame(groups)
    for metric in METRICS:
        weighted = (grouped[metric].fillna(0) * grouped.query_count).sum() / len(q)
        if not np.isclose(weighted, overall[metric], rtol=0, atol=1e-12):
            raise AssertionError("Grouped metrics do not reconstruct aggregate")
    q.to_csv(destination / f"{stem}_queries.csv.gz", index=False, compression="gzip")
    grouped.to_csv(destination / f"{stem}_groups.csv", index=False)
    audit = {"status": "success", "fingerprint": fingerprint, "query_count": len(q),
             "test_queries_sha256": split_hash.hexdigest(), "metrics": overall,
             "original_metrics": {m: row[m] for m in METRICS},
             "max_absolute_difference": max(abs(overall[m] - row[m]) for m in METRICS),
             "runtime_seconds": time.perf_counter() - started,
             "training_performed": False, "torch_threads": torch.get_num_threads(), "prediction_workers": worker_count}
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Verified {stem}: {len(q)} queries, MRR@10={overall['mrr@10']:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="Independent CPU batch workers for neighbourhood models only")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    destination = OUTPUT / "prefix_groups"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(OUTPUT / "final_results.csv")
    frame = frame[frame.scenario.eq("session")]
    if args.models:
        frame = frame[frame.model.isin(args.models)]
    if args.datasets:
        frame = frame[frame.dataset.isin(args.datasets)]
    for row in frame.to_dict("records"):
        replay(row, destination, args.workers)
    print("Requested prefix replays complete", flush=True)


if __name__ == "__main__":
    main()
