from pathlib import Path

import pandas as pd


SOURCE_RESULTS = Path(
    r"C:\Users\manue\Desktop\Uni Unterlagen\MSc\Semester 3\Msc_Arbeit"
    r"\Wöchentliche_Treffen\Recbole_results_zwischenstand_17_06"
    r"\session_full_tuning_results.csv"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "recbole_results"
    / "presentation_summary_2026_06_17"
)


def parse_config(config_json: object, key: str):
    if pd.isna(config_json):
        return pd.NA

    text = str(config_json)
    pattern = f'"{key}":'
    if pattern not in text:
        return pd.NA

    try:
        import json

        return json.loads(text).get(key, pd.NA)
    except Exception:
        return pd.NA


def clean_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[(frame["status"] == "success") & (frame["model"] != "model")]

    numeric_columns = [
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "runtime_seconds",
        "vsknn_k",
        "vsknn_sample_size",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "window_days",
        "decay_lambda",
        "hidden_size",
        "learning_rate",
        "dropout_prob",
        "vsknn_popularity_weight",
        "vstan_popularity_weight",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def add_config_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    config_fields = [
        "vsknn_k",
        "vsknn_sample_size",
        "vsknn_popularity_weight",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "vstan_idf_weighting",
        "vstan_popularity_weight",
        "window_days",
        "decay_lambda",
        "hidden_size",
        "learning_rate",
        "dropout_prob",
        "epochs",
    ]

    for field in config_fields:
        if field not in frame.columns:
            frame[field] = pd.NA
        else:
            frame[field] = frame[field].astype("object")

        missing = frame[field].isna()
        if "config_json" in frame.columns and missing.any():
            frame.loc[missing, field] = frame.loc[missing, "config_json"].apply(
                lambda value: parse_config(value, field)
            )

    return frame


def build_best_per_dataset_model(results: pd.DataFrame) -> pd.DataFrame:
    results = add_config_columns(results)

    best = (
        results.sort_values("mrr@10", ascending=False)
        .drop_duplicates(["dataset", "model"], keep="first")
        .sort_values(["dataset", "mrr@10"], ascending=[True, False])
    )

    columns = [
        "dataset",
        "model",
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "runtime_seconds",
        "vsknn_k",
        "vsknn_sample_size",
        "vsknn_popularity_weight",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "vstan_idf_weighting",
        "vstan_popularity_weight",
        "window_days",
        "decay_lambda",
        "hidden_size",
        "learning_rate",
        "dropout_prob",
        "epochs",
        "config_json",
    ]

    return best[[column for column in columns if column in best.columns]]


def build_weighted_knn_comparison(results: pd.DataFrame) -> pd.DataFrame:
    knn = add_config_columns(results[results["model"].isin(["VS-KNN", "VSTAN"])])

    weight_column = knn.apply(
        lambda row: row["vsknn_popularity_weight"]
        if row["model"] == "VS-KNN"
        else row["vstan_popularity_weight"],
        axis=1,
    )

    knn = knn.assign(popularity_weight=pd.to_numeric(weight_column, errors="coerce"))
    knn["weighting_variant"] = knn["popularity_weight"].apply(
        lambda value: "unweighted" if pd.isna(value) or value == 0 else "weighted"
    )

    best = (
        knn.sort_values("mrr@10", ascending=False)
        .drop_duplicates(["dataset", "model", "weighting_variant"], keep="first")
        .sort_values(["dataset", "model", "weighting_variant"])
    )

    columns = [
        "dataset",
        "model",
        "weighting_variant",
        "popularity_weight",
        "mrr@10",
        "hit@10",
        "ndcg@10",
        "runtime_seconds",
        "vsknn_k",
        "vsknn_sample_size",
        "vstan_k",
        "vstan_sample_size",
        "vstan_position_decay",
        "vstan_idf_weighting",
        "config_json",
    ]

    return best[[column for column in columns if column in best.columns]]


def write_workbook(
    best_per_dataset_model: pd.DataFrame,
    weighted_comparison: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_best = OUTPUT_DIR / "model_performance_best_config_by_dataset.csv"
    csv_weighted = OUTPUT_DIR / "vsknn_vstan_weighted_vs_unweighted.csv"
    xlsx_path = OUTPUT_DIR / "model_performance_summary.xlsx"

    best_per_dataset_model.to_csv(csv_best, index=False)
    weighted_comparison.to_csv(csv_weighted, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        best_per_dataset_model.to_excel(
            writer,
            sheet_name="Best by Dataset Model",
            index=False,
        )
        weighted_comparison.to_excel(
            writer,
            sheet_name="Weighted KNN Compare",
            index=False,
        )

    return xlsx_path


def main() -> None:
    results = clean_results(pd.read_csv(SOURCE_RESULTS))
    best_per_dataset_model = build_best_per_dataset_model(results)
    weighted_comparison = build_weighted_knn_comparison(results)
    xlsx_path = write_workbook(best_per_dataset_model, weighted_comparison)
    print(xlsx_path)


if __name__ == "__main__":
    main()
