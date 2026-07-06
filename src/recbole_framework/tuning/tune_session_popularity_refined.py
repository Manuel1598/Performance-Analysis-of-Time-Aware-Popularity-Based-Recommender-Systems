from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.recbole_framework.custom_models.session.popularity_recbole import (
    SessionDecayPopRecBole,
    SessionMostPopRecBole,
    SessionRecentPopRecBole,
)
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger
from src.recbole_framework.tuning.tune_session_models_full import (
    load_completed_run_ids,
    run_and_store,
)


DATASETS = [
    "adressa_recbole_sample",
    "globo_recbole_sample",
    "yoochoose_recbole_sample",
]

RECENT_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]
DECAY_HALF_LIFE_DAYS = [0.25, 0.5, 1, 3, 7, 14, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run refined session popularity baselines. This isolates the "
            "popularity baselines after the fixed-day RecentPop windows and "
            "tiny decay lambdas were found to collapse to MostPop on several "
            "datasets."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Session datasets to evaluate.",
    )
    parser.add_argument(
        "--recent-fractions",
        nargs="+",
        type=float,
        default=RECENT_FRACTIONS,
        help=(
            "Relative RecentPop windows as fractions of the training time span. "
            "For example, 0.05 means the most recent 5%% of training time."
        ),
    )
    parser.add_argument(
        "--half-life-days",
        nargs="+",
        type=float,
        default=DECAY_HALF_LIFE_DAYS,
        help="DecayPop half-life values in days.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "session_popularity_refined_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "session_popularity_refined_log.csv"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_completed_run_ids(output_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running refined session popularity baselines")
    print("Reason: avoid RecentPop/DecayPop degenerating into MostPop.")
    print("RecentPop uses relative time-span windows.")
    print("DecayPop uses interpretable half-life values in days.")
    print(f"Using device: {device}")
    print(f"Output file: {output_file}")
    print(f"Already completed refined runs: {len(completed_run_ids)}")

    for dataset_name in args.datasets:
        print(f"\n===== DATASET: {dataset_name} =====")

        mostpop_config = {
            "refinement_name": "session_popularity_refined",
            "refinement_reason": "baseline comparison anchor",
        }
        print(f"Running MostPop on {dataset_name}: {mostpop_config}")
        run_and_store(
            output_file=output_file,
            logger=logger,
            completed_run_ids=completed_run_ids,
            model_class=SessionMostPopRecBole,
            model_name="MostPop",
            dataset_name=dataset_name,
            config_updates=mostpop_config,
            device=device,
        )

        for recent_fraction in args.recent_fractions:
            config_updates = {
                "recent_fraction": recent_fraction,
                "refinement_name": "session_popularity_refined",
                "refinement_reason": "relative recent window",
            }
            print(f"Running RecentPop on {dataset_name}: {config_updates}")
            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=SessionRecentPopRecBole,
                model_name="RecentPop",
                dataset_name=dataset_name,
                config_updates=config_updates,
                device=device,
            )

        for half_life_days in args.half_life_days:
            config_updates = {
                "decay_half_life_days": half_life_days,
                "refinement_name": "session_popularity_refined",
                "refinement_reason": "interpretable half-life decay",
            }
            print(f"Running DecayPop on {dataset_name}: {config_updates}")
            run_and_store(
                output_file=output_file,
                logger=logger,
                completed_run_ids=completed_run_ids,
                model_class=SessionDecayPopRecBole,
                model_name="DecayPop",
                dataset_name=dataset_name,
                config_updates=config_updates,
                device=device,
            )

    print(f"\nSaved refined popularity results to: {output_file}")
    print(f"Saved refined popularity log to: {log_file}")


if __name__ == "__main__":
    main()
