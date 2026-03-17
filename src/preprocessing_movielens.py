from pathlib import Path
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "raw" / "movielens" / "ratings.csv"
    output_file = project_root / "data" / "processed" / "movielens_interactions.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please make sure MovieLens ratings.csv is located in data/raw/movielens/."
        )

    print("Loading MovieLens ratings data...")
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

    print("Renaming columns to unified schema...")
    df.columns = ["user_id", "item_id", "timestamp"]

    print("Filtering users with fewer than 2 interactions...")
    user_counts = df["user_id"].value_counts()
    print(f"Users with exactly 1 interaction before filtering: {(user_counts == 1).sum():,}")
    print(f"Minimum interactions per user before filtering: {user_counts.min():,}")

    valid_users = user_counts[user_counts >= 2].index
    removed_users = (user_counts < 2).sum()
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    filtered_user_counts = df["user_id"].value_counts()
    print(f"Users after filtering: {df['user_id'].nunique():,}")
    print(f"Minimum interactions per user after filtering: {filtered_user_counts.min():,}")

    print("Sorting interactions chronologically per user...")
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Saving processed interactions...")
    df.to_csv(output_file, index=False)

    print("\nPreprocessing completed successfully.")
    print(f"Saved file: {output_file}")
    print(f"Number of interactions: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique():,}")
    print(f"Number of items: {df['item_id'].nunique():,}")
    print(f"Removed users with < 2 interactions: {removed_users:,}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()