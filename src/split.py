from pathlib import Path
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "processed" / "movielens_interactions.csv"
    train_file = project_root / "data" / "processed" / "movielens_train.csv"
    test_file = project_root / "data" / "processed" / "movielens_test.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Processed interaction file not found: {input_file}\n"
            "Please run preprocessing_movielens.py first."
        )

    print("Loading processed MovieLens interactions...")
    df = pd.read_csv(input_file)

    required_columns = ["user_id", "item_id", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("Sorting interactions by user and timestamp...")
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    print("Creating leave-one-out split...")
    test_df = df.groupby("user_id", group_keys=False).tail(1).copy()
    train_df = df.drop(test_df.index).copy()

    train_file.parent.mkdir(parents=True, exist_ok=True)

    print("Saving train and test files...")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\nSplit completed successfully.")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")
    print(f"Users in train: {train_df['user_id'].nunique():,}")
    print(f"Users in test: {test_df['user_id'].nunique():,}")

    users_total = df["user_id"].nunique()
    users_test = test_df["user_id"].nunique()

    print(f"Total users: {users_total:,}")
    print(f"Users with exactly one test interaction: {users_test:,}")

    print("\nTrain preview:")
    print(train_df.head())

    print("\nTest preview:")
    print(test_df.head())


if __name__ == "__main__":
    main()