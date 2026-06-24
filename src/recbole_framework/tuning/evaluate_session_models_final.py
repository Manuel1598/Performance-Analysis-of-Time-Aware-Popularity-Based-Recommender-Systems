from pathlib import Path

import pandas as pd
import torch

from recbole.model.sequential_recommender import GRU4Rec

from src.recbole_framework.custom_models.session.popularity_recbole import (
    SessionDecayPopRecBole,
    SessionMostPopRecBole,
    SessionRecentPopRecBole,
)
from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNNRecBole
from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole
from src.recbole_framework.measurement.experiment_logger import ExperimentLogger
from src.recbole_framework.tuning.tune_session_models_full import (
    load_completed_run_ids,
    run_and_store,
)


TOP_CONFIGS_PER_MODEL = 3

SAMPLE_TO_FULL_DATASET = {
    "yoochoose_recbole_sample": "yoochoose_recbole",
    "globo_recbole_sample": "globo_recbole",
    "adressa_recbole_sample": "adressa_recbole",
}

MODEL_CLASSES = {
    "MostPop": SessionMostPopRecBole,
    "RecentPop": SessionRecentPopRecBole,
    "DecayPop": SessionDecayPopRecBole,
    "VS-KNN": VSKNNRecBole,
    "VSTAN": VSTANRecBole,
    "GRU4Rec": GRU4Rec,
}

MODEL_CONFIG_FIELDS = {
    "MostPop": {},
    "RecentPop": {
        "window_days": int,
        "recent_fraction": float,
    },
    "DecayPop": {
        "decay_lambda": float,
        "decay_half_life_days": float,
    },
    "VS-KNN": {
        "vsknn_k": int,
        "vsknn_sample_size": int,
        "vsknn_popularity_weight": float,
    },
    "VSTAN": {
        "vstan_k": int,
        "vstan_sample_size": int,
        "vstan_position_decay": float,
        "vstan_idf_weighting": bool,
        "vstan_popularity_weight": float,
    },
    "GRU4Rec": {
        "hidden_size": int,
        "learning_rate": float,
        "dropout_prob": float,
        "epochs": int,
    },
}


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() == "true"

    return bool(value)


def build_config_updates(model_name: str, row: pd.Series) -> dict:
    config_updates = {}

    for field_name, caster in MODEL_CONFIG_FIELDS[model_name].items():
        if field_name not in row:
            continue

        value = row[field_name]

        if pd.isna(value):
            continue

        if caster is bool:
            config_updates[field_name] = normalize_bool(value)
        else:
            config_updates[field_name] = caster(value)

    if model_name == "GRU4Rec":
        config_updates.update(
            {
                "model": "GRU4Rec",
                "num_layers": 1,
                "loss_type": "CE",
                "train_neg_sample_args": None,
                "train_batch_size": 2048,
                "eval_batch_size": 2048,
            }
        )

    return config_updates


def load_final_configurations(tuning_results_file: Path) -> pd.DataFrame:
    if not tuning_results_file.exists():
        raise FileNotFoundError(
            f"Session tuning results not found: {tuning_results_file}"
        )

    results_df = pd.read_csv(tuning_results_file)

    successful_results = results_df[results_df["status"] == "success"].copy()
    successful_results = successful_results[
        successful_results["dataset"].isin(SAMPLE_TO_FULL_DATASET.keys())
    ]

    if successful_results.empty:
        raise ValueError("No successful sample tuning results found.")

    selected_rows = []

    grouped_results = successful_results.sort_values(
        "mrr@10",
        ascending=False,
    ).groupby(["dataset", "model"], sort=False)

    for _, group_df in grouped_results:
        selected_rows.append(group_df.head(TOP_CONFIGS_PER_MODEL))

    return pd.concat(selected_rows, ignore_index=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]

    tuning_results_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "session_full_tuning_results.csv"
    )

    output_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "session_final_full_evaluation_results.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_file = (
        project_root
        / "recbole_results"
        / "experiment_logs"
        / "session_final_full_evaluation_log.csv"
    )

    logger = ExperimentLogger(log_file)
    completed_run_ids = load_completed_run_ids(output_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Top configs per model: {TOP_CONFIGS_PER_MODEL}")
    print(f"Already completed final runs: {len(completed_run_ids)}")

    final_configurations = load_final_configurations(tuning_results_file)

    for _, row in final_configurations.iterrows():
        sample_dataset_name = row["dataset"]
        full_dataset_name = SAMPLE_TO_FULL_DATASET[sample_dataset_name]
        model_name = row["model"]

        if model_name not in MODEL_CLASSES:
            print(f"Skipping unsupported model in final evaluation: {model_name}")
            continue

        config_updates = build_config_updates(model_name, row)

        print(
            f"Running final full evaluation for {model_name}: "
            f"{sample_dataset_name} -> {full_dataset_name}, {config_updates}"
        )

        run_and_store(
            output_file=output_file,
            logger=logger,
            completed_run_ids=completed_run_ids,
            model_class=MODEL_CLASSES[model_name],
            model_name=model_name,
            dataset_name=full_dataset_name,
            config_updates=config_updates,
            device=device,
        )

    print(f"\nSaved final session evaluation results to: {output_file}")
    print(f"Saved final session evaluation log to: {log_file}")


if __name__ == "__main__":
    main()
