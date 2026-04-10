from pathlib import Path
import pandas as pd

from src.utils.io import load_data, REQUIRED_INTERACTION_COLUMNS


def convert_to_recbole_interaction_format(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert the processed MovieLens interaction data into
    RecBole's atomic interaction format.

    RecBole expects column names with field types, e.g.:
    user_id:token, item_id:token, timestamp:float
    """
    recbole_df = input_df.copy()

    recbole_df = recbole_df.rename(columns={
        "user_id": "user_id:token",
        "item_id": "item_id:token",
        "timestamp": "timestamp:float",
    })

    return recbole_df


def save_recbole_inter_file(
    recbole_df: pd.DataFrame,
    output_file: Path
) -> None:
    print(f"Saving RecBole interaction file to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    recbole_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    input_file = project_root / "data" / "processed" / "movielens_train.csv"
    output_file = (
        project_root
        / "data"
        / "recbole"
        / "movielens_recbole"
        / "movielens_recbole.inter"
    )

    train_df = load_data(
        input_file,
        "MovieLens training data",
        required_columns=REQUIRED_INTERACTION_COLUMNS
    )

    print(f"\nLoaded training interactions: {len(train_df):,}")
    print(f"Users: {train_df['user_id'].nunique():,}")
    print(f"Items: {train_df['item_id'].nunique():,}")

    recbole_df = convert_to_recbole_interaction_format(train_df)

    print("\nPreview of RecBole-formatted interactions:")
    print(recbole_df.head())

    save_recbole_inter_file(recbole_df, output_file)

    print("\nFinished preparing RecBole dataset.")
    print(f"Saved file: {output_file}")


if __name__ == "__main__":
    main()