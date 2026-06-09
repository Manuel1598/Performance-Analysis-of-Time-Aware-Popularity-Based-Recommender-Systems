from pathlib import Path

import pandas as pd


def load_yoochoose_clicks(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Yoochoose clicks file not found: {input_file}")

    return pd.read_csv(
        input_file,
        header=None,
        names=["session_id", "timestamp", "item_id", "category"],
        dtype={
            "session_id": str,
            "timestamp": str,
            "item_id": str,
            "category": str,
        },
        low_memory=False,
    )


def preprocess_yoochoose(df: pd.DataFrame, min_session_length: int = 2) -> pd.DataFrame:
    df = df[["session_id", "item_id", "timestamp"]].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**6).astype("int64")

    df = df.sort_values(["session_id", "timestamp"])

    session_lengths = df.groupby("session_id").size()
    valid_sessions = session_lengths[session_lengths >= min_session_length].index
    df = df[df["session_id"].isin(valid_sessions)].copy()

    df = df.rename(columns={"session_id": "user_id"})

    return df


def sample_sessions(
    df: pd.DataFrame,
    max_interactions: int = 500_000,
) -> pd.DataFrame:
    session_summary = (
        df.groupby("user_id")
        .agg(
            session_length=("item_id", "size"),
            latest_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .sort_values("latest_timestamp", ascending=False)
    )

    session_summary["cumulative_interactions"] = session_summary[
        "session_length"
    ].cumsum()

    selected_sessions = session_summary[
        session_summary["cumulative_interactions"] <= max_interactions
    ]["user_id"]

    sampled_df = df[df["user_id"].isin(selected_sessions)].copy()
    sampled_df = sampled_df.sort_values(["user_id", "timestamp"])

    return sampled_df


def convert_to_recbole_interaction_format(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )


def save_recbole_inter_file(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved RecBole sample file to: {output_file}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]

    input_file = (
        project_root
        / "data"
        / "raw"
        / "yoochoose"
        / "yoochoose-clicks.dat"
    )

    output_file = (
        project_root
        / "data"
        / "recbole"
        / "yoochoose_recbole_sample"
        / "yoochoose_recbole_sample.inter"
    )

    clicks_df = load_yoochoose_clicks(input_file)

    print(f"\nLoaded Yoochoose clicks: {len(clicks_df):,}")
    print(f"Sessions: {clicks_df['session_id'].nunique():,}")
    print(f"Items: {clicks_df['item_id'].nunique():,}")

    processed_df = preprocess_yoochoose(clicks_df)

    print("\nAfter preprocessing:")
    print(f"Interactions: {len(processed_df):,}")
    print(f"Sessions/users: {processed_df['user_id'].nunique():,}")
    print(f"Items: {processed_df['item_id'].nunique():,}")

    sampled_df = sample_sessions(
        processed_df,
        max_interactions=500_000,
    )

    print("\nAfter session sampling:")
    print(f"Interactions: {len(sampled_df):,}")
    print(f"Sessions/users: {sampled_df['user_id'].nunique():,}")
    print(f"Items: {sampled_df['item_id'].nunique():,}")

    recbole_df = convert_to_recbole_interaction_format(sampled_df)

    print("\nPreview:")
    print(recbole_df.head())

    save_recbole_inter_file(recbole_df, output_file)


if __name__ == "__main__":
    main()
