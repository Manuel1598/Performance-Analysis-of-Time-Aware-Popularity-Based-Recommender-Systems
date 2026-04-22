from pathlib import Path
import pandas as pd

from src.prototype.utils.io import load_data, REQUIRED_INTERACTION_COLUMNS


def split_leave_one_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Creating chronological leave-one-out split...")

    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    test_indices = df.groupby("user_id").tail(1).index
    test_df = df.loc[test_indices].copy().reset_index(drop=True)
    train_df = df.drop(test_indices).copy().reset_index(drop=True)

    return train_df, test_df


def validate_split(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    print("\nValidating split...")

    test_user_counts = test_df.groupby("user_id").size()
    if not (test_user_counts == 1).all():
        raise ValueError("Each user must have exactly one test interaction.")

    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())

    if train_users != test_users:
        raise ValueError("Train and test user sets do not match.")

    if len(train_df) + len(test_df) != len(original_df):
        raise ValueError("Train + test size does not match original data size.")

    print("Split validation passed.")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")
    print(f"Users in train: {train_df['user_id'].nunique():,}")
    print(f"Users in test: {test_df['user_id'].nunique():,}")


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    train_file = output_dir / "amazon_train.csv"
    test_file = output_dir / "amazon_test.csv"

    print(f"\nSaving train split to {train_file}...")
    train_df.to_csv(train_file, index=False)

    print(f"Saving test split to {test_file}...")
    test_df.to_csv(test_file, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_file = project_root / "data" / "processed" / "amazon_interactions.csv"
    output_dir = project_root / "data" / "processed"

    df = load_data(
        input_file,
        "Amazon processed interactions",
        required_columns=REQUIRED_INTERACTION_COLUMNS
    )

    print(f"\nLoaded interactions: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")

    train_df, test_df = split_leave_one_out(df)

    validate_split(df, train_df, test_df)

    print("\nTrain preview:")
    print(train_df.head())

    print("\nTest preview:")
    print(test_df.head())

    save_split(train_df, test_df, output_dir)

    max_train = train_df.groupby("user_id")["timestamp"].max()
    test_ts = test_df.set_index("user_id")["timestamp"]

    violations = (max_train > test_ts).sum()
    print("Chronology violations:", violations)

    print("\nDone.")


if __name__ == "__main__":
    main()