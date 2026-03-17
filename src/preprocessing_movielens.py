from pathlib import Path
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "raw" / "movielens" / "ratings.csv"
    output_file = project_root / "data" / "processed" / "movielens_interactions.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please make sure the MovieLens ratings file is located in data/raw/movielens/."
        )

    print("Loading MovieLens data...")
    df = pd.read_csv(input_file)

    required_columns = ["userId", "movieId", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("Selecting relevant columns...")
    df = df[required_columns].copy()

    print("Renaming columns...")
    df.columns = ["user_id", "item_id", "timestamp"]

    print("Sorting chronologically per user...")
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Saving processed file...")
    df.to_csv(output_file, index=False)

    print("\nPreprocessing completed.")
    print(f"Saved to: {output_file}")
    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()