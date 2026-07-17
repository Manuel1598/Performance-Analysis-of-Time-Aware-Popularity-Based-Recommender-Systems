"""Run the audited VSKNN implementation on one or all session datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNN


SESSION_DATASETS = (
    "yoochoose_recbole_sample",
    "globo_recbole_sample",
    "adressa_recbole_sample",
    "yoochoose_recbole",
    "globo_recbole",
    "adressa_recbole",
)
SAMPLE_DATASETS = SESSION_DATASETS[:3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate audited VSKNN with RecBole 1.2.1."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=SESSION_DATASETS,
        default="yoochoose_recbole_sample",
        help="Session dataset to evaluate (default: yoochoose sample).",
    )
    dataset_group.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate Yoochoose, Globo, and Adressa sample datasets in sequence.",
    )
    parser.add_argument("--neighbor-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args(argv)


def build_config(
    project_root: Path,
    dataset_name: str,
    neighbor_size: int,
    sample_size: int,
    device: str,
    show_progress: bool,
) -> Config:
    config_dict = {
        "model": VSKNN,
        "dataset": dataset_name,
        "data_path": str(project_root / "data" / "recbole"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "MAX_ITEM_LIST_LENGTH": 20,
        "epochs": 1,
        "train_batch_size": 2048,
        "eval_batch_size": 1024,
        "topk": [5, 10],
        "metrics": ["Hit", "NDCG", "MRR"],
        "valid_metric": "MRR@10",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "TO",
            "mode": "full",
        },
        "neighbor_size": neighbor_size,
        "sample_size": sample_size,
        "sampling": "recent",
        "similarity": "vec",
        "session_weighting": "div",
        "score_weighting": "div",
        "seed": 42,
        "reproducibility": True,
        "device": device,
        "show_progress": show_progress,
    }
    return Config(model=VSKNN, config_dict=config_dict)


def run_dataset(
    project_root: Path,
    dataset_name: str,
    neighbor_size: int,
    sample_size: int,
    device: str,
    show_progress: bool,
) -> dict:
    print(f"\n=== Audited VSKNN: {dataset_name} ===")
    config = build_config(
        project_root,
        dataset_name,
        neighbor_size,
        sample_size,
        device,
        show_progress,
    )
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = VSKNN(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    started_at = time.perf_counter()
    trainer.fit(train_data, valid_data, saved=False, verbose=False)
    test_result = trainer.evaluate(test_data, load_best_model=False)
    runtime_seconds = round(time.perf_counter() - started_at, 2)

    result = {
        "model": "VSKNN-audited",
        "dataset": dataset_name,
        "seed": 42,
        "device": device,
        "neighbor_size": neighbor_size,
        "sample_size": sample_size,
        "sampling": "recent",
        "similarity": "vec",
        "session_weighting": "div",
        "score_weighting": "div",
        "reference_sessions": len(model.reference_sessions),
        "runtime_seconds": runtime_seconds,
        **dict(test_result),
    }

    output_dir = project_root / "recbole_results" / "vsknn_audited"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_metrics.csv"
    pd.DataFrame([result]).to_csv(output_file, index=False)

    print("Test results:", dict(test_result))
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print("Saved:", output_file)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[4]
    datasets = SAMPLE_DATASETS if args.all_samples else (args.dataset,)

    results = [
        run_dataset(
            project_root=project_root,
            dataset_name=dataset_name,
            neighbor_size=args.neighbor_size,
            sample_size=args.sample_size,
            device=args.device,
            show_progress=args.show_progress,
        )
        for dataset_name in datasets
    ]

    if len(results) > 1:
        summary_file = (
            project_root
            / "recbole_results"
            / "vsknn_audited"
            / "all_session_samples_summary.csv"
        )
        pd.DataFrame(results).to_csv(summary_file, index=False)
        print("\nSaved combined summary:", summary_file)


if __name__ == "__main__":
    main()
