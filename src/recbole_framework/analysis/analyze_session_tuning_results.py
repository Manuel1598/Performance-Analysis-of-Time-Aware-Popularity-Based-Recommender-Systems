from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    input_file = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "yoochoose_session_tuning_results.csv"
    )

    output_dir = (
        project_root
        / "recbole_results"
        / "tuning_results"
        / "analysis_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(input_file)

    print("\nBest configurations overall by MRR@10:")
    print(
        results_df
        .sort_values("mrr@10", ascending=False)
        .head(10)
    )

    best_per_model = (
        results_df
        .sort_values("mrr@10", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )

    print("\nBest configuration per model:")
    print(best_per_model)

    best_per_model_file = output_dir / "best_session_tuning_configurations_per_model.csv"
    best_overall_file = output_dir / "best_session_tuning_configurations_overall.csv"

    best_per_model.to_csv(best_per_model_file, index=False)

    (
        results_df
        .sort_values("mrr@10", ascending=False)
        .head(20)
        .to_csv(best_overall_file, index=False)
    )

    print(f"\nSaved best per-model configurations to: {best_per_model_file}")
    print(f"Saved best overall configurations to: {best_overall_file}")


if __name__ == "__main__":
    main()