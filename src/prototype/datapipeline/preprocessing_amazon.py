import argparse
import json
from pathlib import Path

import pandas as pd


MIN_POSITIVE_RATING = 4.0
MIN_INTERACTIONS_PER_USER = 2


def resolve_amazon_input(raw_directory: Path) -> Path:
    """Find the Video Games review file below the Amazon raw-data folder."""
    candidates = [
        raw_directory / "Video_Games.jsonl",
        raw_directory / "Video_Games.jsonl" / "Video_Games.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Amazon Video Games file not found. Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_amazon_reviews(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Amazon file not found: {file_path}")

    print(f"Loading Amazon data from {file_path}...")

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)

            if (
                "user_id" in record
                and "parent_asin" in record
                and "rating" in record
                and "timestamp" in record
            ):
                rows.append(
                    {
                        "user_id": str(record["user_id"]),
                        "item_id": str(record["parent_asin"]),
                        "rating": float(record["rating"]),
                        "timestamp": int(record["timestamp"]) // 1000,
                    }
                )

            if i % 1_000_000 == 0 and i > 0:
                print(f"Processed {i:,} lines...")

    df = pd.DataFrame(rows)
    return df


def preprocess_amazon(
    reviews: pd.DataFrame,
    min_rating: float = MIN_POSITIVE_RATING,
    min_interactions_per_user: int = MIN_INTERACTIONS_PER_USER,
) -> pd.DataFrame:
    """Convert Amazon reviews into positive implicit interactions."""
    required_columns = ["user_id", "item_id", "rating", "timestamp"]
    missing_columns = [
        column for column in required_columns if column not in reviews
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(reviews.columns)}"
        )
    if min_interactions_per_user < 2:
        raise ValueError("min_interactions_per_user must be at least 2")

    prepared = reviews[required_columns].copy()
    prepared = prepared.dropna(subset=required_columns)
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="raise")
    prepared["timestamp"] = prepared["timestamp"].astype("int64")
    prepared = prepared[prepared["rating"] >= min_rating].copy()

    positive_counts = prepared["user_id"].value_counts()
    valid_users = positive_counts[
        positive_counts >= min_interactions_per_user
    ].index
    prepared = prepared[prepared["user_id"].isin(valid_users)].copy()
    prepared = prepared[["user_id", "item_id", "timestamp"]]
    return prepared.sort_values(
        by=["user_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Amazon Video Games positive implicit interactions."
        )
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=MIN_POSITIVE_RATING,
        help="Minimum rating treated as positive feedback (default: 4.0).",
    )
    return parser.parse_args()


def save_output(df: pd.DataFrame, output_file: Path) -> None:
    print(f"Saving processed data to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    raw_directory = project_root / "data" / "raw" / "amazon"
    input_file = resolve_amazon_input(raw_directory)
    output_file = (
        project_root
        / "data"
        / "processed"
        / "amazon_positive_interactions.csv"
    )

    df = load_amazon_reviews(input_file)

    print("\nLoaded data:")
    print(f"Interactions: {len(df):,}")

    if len(df) == 0:
        raise ValueError(
            "No valid interactions were loaded. "
            "Please check the input field names in the Amazon JSONL file."
        )

    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")

    positive_rows = int((df["rating"] >= args.min_rating).sum())
    original_users = df["user_id"].nunique()
    print(f"Keeping ratings >= {args.min_rating:g} as positive feedback...")
    df = preprocess_amazon(df, min_rating=args.min_rating)

    print("\nAfter preprocessing:")
    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")
    print(f"Ratings meeting threshold before user filtering: {positive_rows:,}")
    print(
        "Users removed with fewer than "
        f"{MIN_INTERACTIONS_PER_USER} positive interactions: "
        f"{original_users - df['user_id'].nunique():,}"
    )

    print("\nPreview:")
    print(df.head())

    save_output(df, output_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
