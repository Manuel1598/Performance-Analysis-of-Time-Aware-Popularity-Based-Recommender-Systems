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

    print("\n--- DEBUG BEFORE SPLIT ---")
    user_counts = df["user_id"].value_counts()
    print(f"Total interactions: {len(df):,}")
    print(f"Total users: {df['user_id'].nunique():,}")
    print(f"Users with exactly 1 interaction: {(user_counts == 1).sum():,}")
    print(f"Minimum interactions per user: {user_counts.min():,}")
    print(f"Maximum interactions per user: {user_counts.max():,}")

    print("\nCreating leave-one-out split...")
    test_df = df.groupby("user_id", group_keys=False).tail(1).copy()
    train_df = df.drop(test_df.index).copy()

    train_file.parent.mkdir(parents=True, exist_ok=True)

    print("Saving train and test files...")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    missing_in_train = test_users - train_users

    print("\n--- SPLIT SUMMARY ---")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")
    print(f"Users in train: {train_df['user_id'].nunique():,}")
    print(f"Users in test: {test_df['user_id'].nunique():,}")
    print(f"Users missing in train: {len(missing_in_train):,}")

    if missing_in_train:
        print(f"Sample missing user_ids: {list(missing_in_train)[:10]}")

    print("\nTrain preview:")
    print(train_df.head())

    print("\nTest preview:")
    print(test_df.head())


if __name__ == "__main__":
    main()