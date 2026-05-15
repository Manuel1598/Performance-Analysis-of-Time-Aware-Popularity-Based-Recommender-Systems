from pathlib import Path

import pandas as pd


def load_globo_clicks(clicks_dir: Path) -> pd.DataFrame:
    if not clicks_dir.exists():
        raise FileNotFoundError(f"Globo clicks directory not found: {clicks_dir}")

    click_files = sorted(clicks_dir.glob("clicks_hour_*.csv"))

    if not click_files:
        raise FileNotFoundError(f"No clicks_hour_*.csv files found in {clicks_dir}")

    print(f"Found {len(click_files)} Globo click files.")

    dataframes = []

    for file_path in click_files:
        print(f"Loading {file_path.name}...")

        df = pd.read_csv(
            file_path,
            usecols=[
                "session_id",
                "click_article_id",
                "click_timestamp",
            ],
        )

        dataframes.append(df)

    clicks_df = pd.concat(dataframes, ignore_index=True)

    return clicks_df


def preprocess_globo(
    df: pd.DataFrame,
    min_session_length: int = 2,
) -> pd.DataFrame:
    print("Preprocessing Globo clicks...")

    df = df.rename(
        columns={
            "session_id": "user_id",
            "click_article_id": "item_id",
            "click_timestamp": "timestamp",
        }
    )

    df = df[["user_id", "item_id", "timestamp"]].copy()

    # Globo timestamps are in milliseconds.
    df["timestamp"] = (df["timestamp"] // 1000).astype("int64")

    df = df.sort_values(["user_id", "timestamp"])

    session_lengths = df.groupby("user_id").size()
    valid_sessions = session_lengths[session_lengths >= min_session_length].index

    df = df[df["user_id"].isin(valid_sessions)].copy()

    return df


def convert_to_recbole_interaction_format(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )


def sample_sessions(
    df: pd.DataFrame,
    max_interactions: int = 500_000,
    random_state: int = 42,
) -> pd.DataFrame:
    session_sizes = df.groupby("user_id").size().reset_index(name="session_length")

    sampled_sessions = session_sizes.sample(
        frac=1.0,
        random_state=random_state,
    )

    sampled_sessions["cumulative_interactions"] = sampled_sessions[
        "session_length"
    ].cumsum()

    selected_sessions = sampled_sessions[
        sampled_sessions["cumulative_interactions"] <= max_interactions
    ]["user_id"]

    sampled_df = df[df["user_id"].isin(selected_sessions)].copy()
    sampled_df = sampled_df.sort_values(["user_id", "timestamp"])

    return sampled_df


def save_recbole_inter_file(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, sep="\t", index=False)

    print(f"Saved RecBole file to: {output_file}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]

    clicks_dir = (
        project_root
        / "data"
        / "raw"
        / "globo"
        / "clicks"
        / "clicks"
    )

    full_output_file = (
        project_root
        / "data"
        / "recbole"
        / "globo_recbole"
        / "globo_recbole.inter"
    )

    sample_output_file = (
        project_root
        / "data"
        / "recbole"
        / "globo_recbole_sample"
        / "globo_recbole_sample.inter"
    )

    clicks_df = load_globo_clicks(clicks_dir)

    print("\nLoaded Globo clicks:")
    print(f"Interactions: {len(clicks_df):,}")
    print(f"Sessions: {clicks_df['session_id'].nunique():,}")
    print(f"Items: {clicks_df['click_article_id'].nunique():,}")

    processed_df = preprocess_globo(clicks_df)

    print("\nAfter preprocessing:")
    print(f"Interactions: {len(processed_df):,}")
    print(f"Sessions/users: {processed_df['user_id'].nunique():,}")
    print(f"Items: {processed_df['item_id'].nunique():,}")

    recbole_full_df = convert_to_recbole_interaction_format(processed_df)

    print("\nPreview full RecBole format:")
    print(recbole_full_df.head())

    save_recbole_inter_file(recbole_full_df, full_output_file)

    sampled_df = sample_sessions(
        processed_df,
        max_interactions=500_000,
        random_state=42,
    )

    print("\nAfter session sampling:")
    print(f"Interactions: {len(sampled_df):,}")
    print(f"Sessions/users: {sampled_df['user_id'].nunique():,}")
    print(f"Items: {sampled_df['item_id'].nunique():,}")

    recbole_sample_df = convert_to_recbole_interaction_format(sampled_df)

    print("\nPreview sample RecBole format:")
    print(recbole_sample_df.head())

    save_recbole_inter_file(recbole_sample_df, sample_output_file)


if __name__ == "__main__":
    main()