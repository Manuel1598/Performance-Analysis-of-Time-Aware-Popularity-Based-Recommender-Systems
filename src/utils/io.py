from pathlib import Path
import pandas as pd


REQUIRED_INTERACTION_COLUMNS = ["user_id", "item_id", "timestamp"]


def load_data(
    file_path: Path,
    file_description: str,
    required_columns: list[str] | None = None
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description} file not found: {file_path}"
        )

    print(f"Loading {file_description}...")
    df = pd.read_csv(file_path)

    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_description}: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

    return df


def save_recommendations(
    recommendations: dict[int, list[int]],
    output_file: Path
) -> pd.DataFrame:
    print(f"Saving recommendations to {output_file}...")

    rows = []

    for user_id, items in recommendations.items():
        for rank, item_id in enumerate(items, start=1):
            rows.append({
                "user_id": user_id,
                "rank": rank,
                "item_id": item_id
            })

    recommendations_df = pd.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    recommendations_df.to_csv(output_file, index=False)

    return recommendations_df


def save_results(results: dict[str, float], output_file: Path) -> None:
    print(f"Saving evaluation results to {output_file}...")

    df = pd.DataFrame([results])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)