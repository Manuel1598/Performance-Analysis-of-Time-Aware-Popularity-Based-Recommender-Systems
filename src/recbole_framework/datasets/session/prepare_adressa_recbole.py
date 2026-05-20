import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]

    raw_data_dir = (
        project_root
        / "data"
        / "raw"
        / "adressa"
        / "three_month"
    )

    output_dir_full = (
        project_root
        / "data"
        / "recbole"
        / "adressa_recbole"
    )

    output_dir_sample = (
        project_root
        / "data"
        / "recbole"
        / "adressa_recbole_sample"
    )

    output_dir_full.mkdir(parents=True, exist_ok=True)
    output_dir_sample.mkdir(parents=True, exist_ok=True)

    interactions = []

    files = sorted(raw_data_dir.iterdir())

    print(f"Found {len(files)} Adressa files")

    for file_path in files:
        if not file_path.is_file():
            continue

        print(f"Processing: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    event = json.loads(line)

                    if (
                        "userId" not in event
                        or "time" not in event
                        or "id" not in event
                    ):
                        continue

                    user_id = event["userId"]
                    item_id = event["id"]
                    timestamp = int(event["time"])

                    interactions.append(
                        {
                            "user_id": str(user_id),
                            "item_id": str(item_id),
                            "timestamp": timestamp,
                        }
                    )

                except Exception:
                    continue

    interactions_df = pd.DataFrame(interactions)

    print("\nLoaded Adressa interactions:")
    print(f"Interactions: {len(interactions_df):,}")
    print(f"Users: {interactions_df['user_id'].nunique():,}")
    print(f"Items: {interactions_df['item_id'].nunique():,}")

    print("\nPreprocessing interactions...")

    user_counts = interactions_df["user_id"].value_counts()

    valid_users = user_counts[user_counts >= 2].index

    interactions_df = interactions_df[
        interactions_df["user_id"].isin(valid_users)
    ]

    interactions_df = interactions_df.sort_values("timestamp")

    print("\nAfter preprocessing:")
    print(f"Interactions: {len(interactions_df):,}")
    print(f"Users: {interactions_df['user_id'].nunique():,}")
    print(f"Items: {interactions_df['item_id'].nunique():,}")

    recbole_df = pd.DataFrame({
        "user_id:token": interactions_df["user_id"],
        "item_id:token": interactions_df["item_id"],
        "timestamp:float": interactions_df["timestamp"],
    })

    print("\nPreview full RecBole format:")
    print(recbole_df.head())

    full_output_file = (
        output_dir_full
        / "adressa_recbole.inter"
    )

    recbole_df.to_csv(
        full_output_file,
        sep="\t",
        index=False,
    )

    print(f"\nSaved full RecBole file to: {full_output_file}")

    sample_size = 500_000

    sample_df = recbole_df.head(sample_size)

    print("\nAfter sample creation:")
    print(f"Interactions: {len(sample_df):,}")
    print(f"Users: {sample_df['user_id:token'].nunique():,}")
    print(f"Items: {sample_df['item_id:token'].nunique():,}")

    print("\nPreview sample RecBole format:")
    print(sample_df.head())

    sample_output_file = (
        output_dir_sample
        / "adressa_recbole_sample.inter"
    )

    sample_df.to_csv(
        sample_output_file,
        sep="\t",
        index=False,
    )

    print(f"\nSaved sample RecBole file to: {sample_output_file}")


if __name__ == "__main__":
    main()