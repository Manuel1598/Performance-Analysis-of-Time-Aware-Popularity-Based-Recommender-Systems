from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["user_id", "item_id", "timestamp"]


def load_amazon_interactions(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Amazon interactions file not found: {input_file}")

    df = pd.read_csv(input_file)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df[REQUIRED_COLUMNS].copy()


def convert_to_recbole_interaction_format(input_df: pd.DataFrame) -> pd.DataFrame:
    recbole_df = input_df.copy()

    recbole_df = recbole_df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )

    return recbole_df


def save_recbole_inter_file(recbole_df: pd.DataFrame, output_file: Path) -> None:
    print(f"Saving RecBole interaction file to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    recbole_df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    input_file = project_root / "data" / "processed" / "amazon_interactions.csv"
    output_file = (
        project_root
        / "data"
        / "recbole"
        / "amazon_recbole"
        / "amazon_recbole.inter"
    )

    amazon_df = load_amazon_interactions(input_file)

    print(f"\nLoaded Amazon interactions: {len(amazon_df):,}")
    print(f"Users: {amazon_df['user_id'].nunique():,}")
    print(f"Items: {amazon_df['item_id'].nunique():,}")

    amazon_df = amazon_df.sort_values(["user_id", "timestamp"])

    recbole_df = convert_to_recbole_interaction_format(amazon_df)

    print("\nPreview of RecBole-formatted interactions:")
    print(recbole_df.head())

    save_recbole_inter_file(recbole_df, output_file)

    print("\nFinished preparing Amazon RecBole dataset.")
    print(f"Saved file: {output_file}")


if __name__ == "__main__":
    main()