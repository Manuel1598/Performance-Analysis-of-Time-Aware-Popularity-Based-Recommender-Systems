from pathlib import Path
import json
import pandas as pd


def load_amazon_reviews(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Amazon file not found: {file_path}")

    print(f"Loading Amazon data from {file_path}...")

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)

            if i == 0:
                print("\nSAMPLE RECORD:")
                print(record)

            if (
                "user_id" in record
                and "parent_asin" in record
                and "timestamp" in record
            ):
                rows.append({
                    "user_id": str(record["user_id"]),
                    "item_id": str(record["parent_asin"]),
                    "timestamp": int(record["timestamp"]) // 1000,
                })

            if i % 1_000_000 == 0 and i > 0:
                print(f"Processed {i:,} lines...")

    df = pd.DataFrame(rows)
    return df


def preprocess_amazon(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing Amazon interactions...")

    df = df.copy()

    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")


    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)


    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= 2].index
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)

    return df


def save_output(df: pd.DataFrame, output_file: Path) -> None:
    print(f"Saving processed data to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_file = project_root / "data" / "raw" / "amazon" / "Video_Games.jsonl" / "Video_Games.jsonl"
    output_file = project_root / "data" / "processed" / "amazon_interactions.csv"

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

    df = preprocess_amazon(df)

    print("\nAfter preprocessing:")
    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")

    print("\nPreview:")
    print(df.head())

    save_output(df, output_file)

    print("\nDone.")


if __name__ == "__main__":
    main()