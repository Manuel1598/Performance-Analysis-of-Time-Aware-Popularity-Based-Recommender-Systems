import argparse
from pathlib import Path

import pandas as pd


MIN_POSITIVE_RATING = 4.0
MIN_INTERACTIONS_PER_USER = 2


def resolve_movielens_input(raw_directory: Path) -> Path:
    """Find ratings.csv below the MovieLens raw-data folder."""
    direct_file = raw_directory / "ratings.csv"
    if direct_file.is_file():
        return direct_file

    nested_files = sorted(raw_directory.glob("*/ratings.csv"))
    if len(nested_files) == 1:
        return nested_files[0]
    if len(nested_files) > 1:
        raise FileNotFoundError(
            "Multiple MovieLens ratings.csv files found below "
            f"{raw_directory}: {nested_files}"
        )
    raise FileNotFoundError(
        "MovieLens ratings.csv not found. Expected it directly in "
        f"{raw_directory} or in one immediate subdirectory."
    )


def prepare_movielens_interactions(
    ratings: pd.DataFrame,
    min_rating: float = MIN_POSITIVE_RATING,
    min_interactions_per_user: int = MIN_INTERACTIONS_PER_USER,
) -> pd.DataFrame:
    """Convert explicit MovieLens ratings into positive implicit interactions.

    Only ratings at or above ``min_rating`` are treated as positive feedback.
    Users are retained only when they have enough positive interactions after
    this threshold has been applied. The returned frame contains exactly the
    three fields consumed by the RecBole Top-N pipeline.
    """
    required_columns = ["userId", "movieId", "rating", "timestamp"]
    missing_columns = [column for column in required_columns if column not in ratings]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(ratings.columns)}"
        )
    if min_interactions_per_user < 2:
        raise ValueError("min_interactions_per_user must be at least 2")

    prepared = ratings[required_columns].copy()
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="raise")
    prepared = prepared[prepared["rating"] >= min_rating].copy()

    positive_counts = prepared["userId"].value_counts()
    valid_users = positive_counts[
        positive_counts >= min_interactions_per_user
    ].index
    prepared = prepared[prepared["userId"].isin(valid_users)].copy()

    prepared = prepared.rename(
        columns={"userId": "user_id", "movieId": "item_id"}
    )
    prepared = prepared[["user_id", "item_id", "timestamp"]]
    return prepared.sort_values(
        by=["user_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MovieLens positive implicit interactions for RecBole."
        )
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=MIN_POSITIVE_RATING,
        help="Minimum rating treated as positive feedback (default: 4.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    raw_directory = project_root / "data" / "raw" / "movielens"
    input_file = resolve_movielens_input(raw_directory)
    output_file = (
        project_root
        / "data"
        / "processed"
        / "movielens_positive_interactions.csv"
    )

    print(f"Loading MovieLens ratings data from {input_file}...")
    ratings = pd.read_csv(input_file)
    total_users = ratings["userId"].nunique() if "userId" in ratings else 0

    print(f"Keeping ratings >= {args.min_rating:g} as positive feedback...")
    positive_rows = 0
    if "rating" in ratings:
        numeric_ratings = pd.to_numeric(ratings["rating"], errors="raise")
        positive_rows = int((numeric_ratings >= args.min_rating).sum())
    df = prepare_movielens_interactions(ratings, min_rating=args.min_rating)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Saving processed interactions...")
    df.to_csv(output_file, index=False)

    print("\nPreprocessing completed successfully.")
    print(f"Saved file: {output_file}")
    print(f"Number of interactions: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique():,}")
    print(f"Number of items: {df['item_id'].nunique():,}")
    print(f"Ratings meeting threshold before user filtering: {positive_rows:,}")
    print(
        "Users removed with fewer than "
        f"{MIN_INTERACTIONS_PER_USER} positive interactions: "
        f"{total_users - df['user_id'].nunique():,}"
    )
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
