from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import torch
from recbole.model.general_recommender import BPR
from recbole.model.sequential_recommender import GRU4Rec

from src.recbole_framework.custom_models.session.popularity_recbole import (
    SessionDecayPopRecBole,
    SessionMostPopRecBole,
    SessionRecentPopRecBole,
)
from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole
from src.recbole_framework.custom_models.topn.decaypop_recbole import DecayPopRecBole
from src.recbole_framework.custom_models.topn.mostpop_recbole import MostPopRecBole
from src.recbole_framework.custom_models.topn.recentpop_recbole import RecentPopRecBole
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger
from src.recbole_framework.tuning.tune_session_models_full import (
    load_completed_run_ids as load_session_completed_run_ids,
    run_and_store as run_session_and_store,
)
from src.recbole_framework.tuning.tune_topn_models_full import (
    load_completed_run_ids as load_topn_completed_run_ids,
    run_and_store as run_topn_and_store,
)


TOPN_DATASETS = ["movielens_recbole", "amazon_recbole"]
SESSION_DATASETS = ["adressa_recbole", "globo_recbole", "yoochoose_recbole"]

TOPN_MODELS = ["MostPop", "RecentPop", "DecayPop"]
SESSION_MODELS = ["MostPop", "RecentPop", "DecayPop", "VS-KNN", "VSTAN", "GRU4Rec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full server experiments in Docker. Outputs are separate from "
            "local tuning files and can be resumed after interruption."
        )
    )
    parser.add_argument("--skip-topn", action="store_true", help="Do not run Top-N experiments.")
    parser.add_argument("--skip-session", action="store_true", help="Do not run session experiments.")
    parser.add_argument(
        "--topn-datasets",
        nargs="+",
        default=TOPN_DATASETS,
        help="Top-N datasets to run.",
    )
    parser.add_argument(
        "--session-datasets",
        nargs="+",
        default=SESSION_DATASETS,
        help="Full session datasets to run.",
    )
    parser.add_argument(
        "--topn-models",
        nargs="+",
        default=TOPN_MODELS,
        help="Top-N models to run. Use --include-bpr to add BPR.",
    )
    parser.add_argument(
        "--session-models",
        nargs="+",
        default=SESSION_MODELS,
        help="Session models to run.",
    )
    parser.add_argument(
        "--include-bpr",
        action="store_true",
        help="Also run BPR for Top-N experiments.",
    )
    parser.add_argument(
        "--output-prefix",
        default="server_full",
        help="Prefix for result and log CSV files.",
    )
    parser.add_argument("--topn-recent-window-days", nargs="+", type=int, default=[1, 3, 7, 14, 30, 60, 90, 180])
    parser.add_argument("--topn-decay-lambdas", nargs="+", type=float, default=[1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6])
    parser.add_argument("--session-recent-fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--session-decay-half-life-days", nargs="+", type=float, default=[0.25, 0.5, 1, 3, 7, 14, 30])
    parser.add_argument("--knn-k-values", nargs="+", type=int, default=[100, 200, 500])
    parser.add_argument("--knn-sample-sizes", nargs="+", type=int, default=[500, 1000, 2000, 5000])
    parser.add_argument("--popularity-weights", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--vstan-position-decays", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--gru-hidden-sizes", nargs="+", type=int, default=[100, 128, 256])
    parser.add_argument("--gru-learning-rates", nargs="+", type=float, default=[0.001, 0.0005, 0.0001])
    parser.add_argument("--gru-dropout-probs", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--gru-epochs", nargs="+", type=int, default=[10, 20])
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def warn_about_missing_datasets(dataset_names: list[str]) -> None:
    data_root = project_root() / "data" / "recbole"
    for dataset_name in dataset_names:
        dataset_dir = data_root / dataset_name
        if not dataset_dir.exists():
            print(f"WARNING: dataset directory not found: {dataset_dir}")


def run_topn_experiments(args: argparse.Namespace, device: str) -> None:
    output_file = (
        project_root()
        / "recbole_results"
        / "tuning_results"
        / f"{args.output_prefix}_topn_results.csv"
    )
    log_file = (
        project_root()
        / "recbole_results"
        / "experiment_logs"
        / f"{args.output_prefix}_topn_log.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_topn_completed_run_ids(output_file)
    selected_models = set(args.topn_models)
    if args.include_bpr:
        selected_models.add("BPR")

    print("\n===== TOP-N SERVER EXPERIMENTS =====")
    print(f"Output file: {output_file}")
    print(f"Already completed Top-N runs: {len(completed_run_ids)}")
    warn_about_missing_datasets(args.topn_datasets)

    for dataset_name in args.topn_datasets:
        print(f"\n===== TOP-N DATASET: {dataset_name} =====")

        if "MostPop" in selected_models:
            run_topn_and_store(
                output_file,
                logger,
                completed_run_ids,
                MostPopRecBole,
                "MostPop",
                dataset_name,
                {},
                device,
            )

        if "RecentPop" in selected_models:
            for window_days in args.topn_recent_window_days:
                run_topn_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    RecentPopRecBole,
                    "RecentPop",
                    dataset_name,
                    {"window_days": window_days},
                    device,
                )

        if "DecayPop" in selected_models:
            for decay_lambda in args.topn_decay_lambdas:
                run_topn_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    DecayPopRecBole,
                    "DecayPop",
                    dataset_name,
                    {"decay_lambda": decay_lambda},
                    device,
                )

        if "BPR" in selected_models:
            for embedding_size, learning_rate, epochs in product(
                [32, 64, 128],
                [0.001, 0.0005, 0.0001],
                [50],
            ):
                run_topn_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    BPR,
                    "BPR",
                    dataset_name,
                    {
                        "embedding_size": embedding_size,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "train_neg_sample_args": {
                            "distribution": "uniform",
                            "sample_num": 1,
                            "alpha": 1.0,
                            "dynamic": False,
                            "candidate_num": 0,
                        },
                    },
                    device,
                )


def run_session_experiments(args: argparse.Namespace, device: str) -> None:
    output_file = (
        project_root()
        / "recbole_results"
        / "tuning_results"
        / f"{args.output_prefix}_session_results.csv"
    )
    log_file = (
        project_root()
        / "recbole_results"
        / "experiment_logs"
        / f"{args.output_prefix}_session_log.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_session_completed_run_ids(output_file)
    selected_models = set(args.session_models)

    print("\n===== SESSION SERVER EXPERIMENTS =====")
    print(f"Output file: {output_file}")
    print(f"Already completed session runs: {len(completed_run_ids)}")
    warn_about_missing_datasets(args.session_datasets)

    for dataset_name in args.session_datasets:
        print(f"\n===== SESSION DATASET: {dataset_name} =====")

        if "MostPop" in selected_models:
            run_session_and_store(
                output_file,
                logger,
                completed_run_ids,
                SessionMostPopRecBole,
                "MostPop",
                dataset_name,
                {},
                device,
            )

        if "RecentPop" in selected_models:
            for recent_fraction in args.session_recent_fractions:
                run_session_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    SessionRecentPopRecBole,
                    "RecentPop",
                    dataset_name,
                    {"recent_fraction": recent_fraction},
                    device,
                )

        if "DecayPop" in selected_models:
            for half_life_days in args.session_decay_half_life_days:
                run_session_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    SessionDecayPopRecBole,
                    "DecayPop",
                    dataset_name,
                    {"decay_half_life_days": half_life_days},
                    device,
                )

        if "VS-KNN" in selected_models:
            for k, sample_size, popularity_weight in product(
                args.knn_k_values,
                args.knn_sample_sizes,
                args.popularity_weights,
            ):
                run_session_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    VSKNNRecBole,
                    "VS-KNN",
                    dataset_name,
                    {
                        "vsknn_k": k,
                        "vsknn_sample_size": sample_size,
                        "vsknn_popularity_weight": popularity_weight,
                    },
                    device,
                )

        if "VSTAN" in selected_models:
            for k, sample_size, position_decay, idf_weighting, popularity_weight in product(
                args.knn_k_values,
                args.knn_sample_sizes,
                args.vstan_position_decays,
                [True, False],
                args.popularity_weights,
            ):
                run_session_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    VSTANRecBole,
                    "VSTAN",
                    dataset_name,
                    {
                        "vstan_k": k,
                        "vstan_sample_size": sample_size,
                        "vstan_position_decay": position_decay,
                        "vstan_idf_weighting": idf_weighting,
                        "vstan_popularity_weight": popularity_weight,
                    },
                    device,
                )

        if "GRU4Rec" in selected_models:
            for hidden_size, learning_rate, dropout_prob, epochs in product(
                args.gru_hidden_sizes,
                args.gru_learning_rates,
                args.gru_dropout_probs,
                args.gru_epochs,
            ):
                run_session_and_store(
                    output_file,
                    logger,
                    completed_run_ids,
                    GRU4Rec,
                    "GRU4Rec",
                    dataset_name,
                    {
                        "model": "GRU4Rec",
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "dropout_prob": dropout_prob,
                        "epochs": epochs,
                        "num_layers": 1,
                        "loss_type": "CE",
                        "train_neg_sample_args": None,
                        "train_batch_size": 2048,
                        "eval_batch_size": 2048,
                    },
                    device,
                )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not args.skip_topn:
        run_topn_experiments(args, device)

    if not args.skip_session:
        run_session_experiments(args, device)


if __name__ == "__main__":
    main()
