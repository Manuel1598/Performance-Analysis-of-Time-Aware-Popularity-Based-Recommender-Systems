from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    input_file = (
            project_root
            / "recbole_results"
            / "tuning_results"
            / "session_full_tuning_results.csv"
    )

    output_dir = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "analysis_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(input_file)

    successful_results = results_df[results_df["status"] == "success"].copy()

    summary_columns = [
        "dataset",
        "model",
        "hit@10",
        "ndcg@10",
        "mrr@10",
        "coverage@10",
        "avg_recommendation_popularity@10",
        "runtime_seconds",
        "train_runtime_seconds",
        "eval_runtime_seconds",
        "extra_metrics_runtime_seconds",
        "vsknn_k",
        "vsknn_sample_size",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "vstan_idf_weighting",
        "hidden_size",
        "learning_rate",
        "dropout_prob",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "device",
        "status",
    ]

    available_columns = [
        col for col in summary_columns
        if col in successful_results.columns
    ]

    best_overall = (
        successful_results
        .sort_values("mrr@10", ascending=False)
        .head(20)
    )

    best_per_model = (
        successful_results
        .sort_values("mrr@10", ascending=False)
        .groupby(["dataset", "model"], as_index=False)
        .first()
    )

    print("\nBest configurations overall by MRR@10:")
    print(best_overall[available_columns].head(10))

    print("\nBest configuration per model:")
    print(best_per_model[available_columns])

    output_best_overall = (
        output_dir
        / "best_session_tuning_configurations_overall.csv"
    )

    output_best_per_model = (
        output_dir
        / "best_session_tuning_configurations_per_model.csv"
    )

    best_overall[available_columns].to_csv(
        output_best_overall,
        index=False,
    )

    best_per_model[available_columns].to_csv(
        output_best_per_model,
        index=False,
    )

    print(f"\nSaved best overall configurations to: {output_best_overall}")
    print(f"Saved best per-model configurations to: {output_best_per_model}")


if __name__ == "__main__":
    main()
