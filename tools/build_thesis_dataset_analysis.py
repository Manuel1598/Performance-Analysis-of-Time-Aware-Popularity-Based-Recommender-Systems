from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.pdfgen import canvas


DATASETS = (
    ("movielens_recbole", "MovieLens", "User history"),
    ("amazon_recbole", "Amazon", "User history"),
    ("adressa_recbole_sample", "Adressa", "Sequence"),
    ("globo_recbole_sample", "Globo", "Session"),
    ("yoochoose_recbole_sample", "Yoochoose", "Session"),
)

SERIES_COLORS = (
    colors.HexColor("#2457A6"),
    colors.HexColor("#C44E52"),
    colors.HexColor("#2E8B57"),
    colors.HexColor("#7A4EAB"),
    colors.HexColor("#C07A00"),
)
LINE_DASHES = (None, (7, 3), (2, 2), (9, 3, 2, 3), (1, 3))


@dataclass
class DatasetProfile:
    dataset: str
    label: str
    entity_type: str
    interactions: int
    entity_counts: np.ndarray
    item_counts: np.ndarray
    timestamp_min: float
    timestamp_max: float
    hourly_counts: Counter[int]


def logical_column(columns: list[str], name: str) -> str:
    for column in columns:
        if column.split(":", maxsplit=1)[0] == name:
            return column
    raise KeyError(f"Missing RecBole column {name!r}: {columns}")


def read_profile(data_root: Path, dataset: str, label: str, entity_type: str) -> DatasetProfile:
    path = data_root / dataset / f"{dataset}.inter"
    if not path.exists():
        raise FileNotFoundError(path)

    entity_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    hourly_counter: Counter[int] = Counter()
    interactions = 0
    timestamp_min = math.inf
    timestamp_max = -math.inf

    for chunk in pd.read_csv(path, sep="\t", chunksize=1_000_000):
        columns = list(chunk.columns)
        entity_col = logical_column(columns, "user_id")
        item_col = logical_column(columns, "item_id")
        timestamp_col = logical_column(columns, "timestamp")

        entity_counter.update(chunk[entity_col].astype(str).value_counts().to_dict())
        item_counter.update(chunk[item_col].astype(str).value_counts().to_dict())

        timestamps = pd.to_numeric(chunk[timestamp_col], errors="coerce").dropna()
        if not timestamps.empty:
            timestamp_min = min(timestamp_min, float(timestamps.min()))
            timestamp_max = max(timestamp_max, float(timestamps.max()))
            hours = np.floor(timestamps.to_numpy(dtype=float) / 3600.0).astype(np.int64)
            unique_hours, counts = np.unique(hours, return_counts=True)
            hourly_counter.update(dict(zip(unique_hours.tolist(), counts.tolist())))

        interactions += len(chunk)

    return DatasetProfile(
        dataset=dataset,
        label=label,
        entity_type=entity_type,
        interactions=interactions,
        entity_counts=np.asarray(list(entity_counter.values()), dtype=np.int64),
        item_counts=np.asarray(list(item_counter.values()), dtype=np.int64),
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        hourly_counts=hourly_counter,
    )


def gini(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(float))
    if ordered.size == 0 or ordered.sum() == 0:
        return 0.0
    n = ordered.size
    indices = np.arange(1, n + 1, dtype=float)
    return float(np.sum((2 * indices - n - 1) * ordered) / (n * ordered.sum()))


def top_share(values: np.ndarray, n: int) -> float:
    ordered = np.sort(values)[::-1]
    return float(ordered[:n].sum() / ordered.sum())


def statistics_row(profile: DatasetProfile) -> dict[str, object]:
    entity = profile.entity_counts
    item = profile.item_counts
    top_one_percent_n = max(1, int(round(item.size * 0.01)))
    return {
        "dataset": profile.dataset,
        "label": profile.label,
        "entity_type": profile.entity_type,
        "interactions": profile.interactions,
        "entities": entity.size,
        "items": item.size,
        "mean_history_length": float(entity.mean()),
        "median_history_length": float(np.quantile(entity, 0.50)),
        "p90_history_length": float(np.quantile(entity, 0.90)),
        "p99_history_length": float(np.quantile(entity, 0.99)),
        "maximum_history_length": int(entity.max()),
        "share_length_two": float(np.mean(entity == 2)),
        "top1_item_share": top_share(item, 1),
        "top10_item_share": top_share(item, 10),
        "top1_percent_item_share": top_share(item, top_one_percent_n),
        "item_popularity_gini": gini(item),
        "timestamp_start_utc": datetime.fromtimestamp(
            profile.timestamp_min, tz=timezone.utc
        ).date().isoformat(),
        "timestamp_end_utc": datetime.fromtimestamp(
            profile.timestamp_max, tz=timezone.utc
        ).date().isoformat(),
        "time_span_days": (profile.timestamp_max - profile.timestamp_min) / 86400.0,
    }


def downsample_xy(x: np.ndarray, y: np.ndarray, maximum: int = 500) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= maximum:
        return x, y
    indices = np.unique(np.linspace(0, x.size - 1, maximum).astype(int))
    return x[indices], y[indices]


def draw_header(c: canvas.Canvas, title: str, width: float, height: float) -> None:
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.black)
    c.drawString(42, height - 28, title)
    c.setFont("Helvetica", 7.5)
    c.drawRightString(width - 28, height - 27, "Processed evaluation datasets")


def draw_legend(c: canvas.Canvas, profiles: list[DatasetProfile], x: float, y: float) -> None:
    c.setFont("Helvetica", 7.5)
    for index, profile in enumerate(profiles):
        row_y = y - index * 13
        c.setStrokeColor(SERIES_COLORS[index])
        c.setLineWidth(1.5)
        if LINE_DASHES[index]:
            c.setDash(LINE_DASHES[index])
        else:
            c.setDash()
        c.line(x, row_y, x + 22, row_y)
        c.setDash()
        c.setFillColor(colors.black)
        c.drawString(x + 28, row_y - 2.5, profile.label)


def draw_axis_labels(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    plot_width: float,
    plot_height: float,
    x_label: str,
    y_label: str,
) -> None:
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    c.drawCentredString(left + plot_width / 2, bottom - 33, x_label)
    c.saveState()
    c.translate(left - 42, bottom + plot_height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()


def draw_log_axes(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    plot_width: float,
    plot_height: float,
    x_max: float,
    y_min: float,
) -> tuple[callable, callable]:
    x_min = 1.0
    x_log_min, x_log_max = math.log10(x_min), math.log10(x_max)
    y_log_min, y_log_max = math.log10(y_min), 0.0

    def sx(value: float) -> float:
        return left + (math.log10(value) - x_log_min) / (x_log_max - x_log_min) * plot_width

    def sy(value: float) -> float:
        return bottom + (math.log10(value) - y_log_min) / (y_log_max - y_log_min) * plot_height

    c.setFont("Helvetica", 7)
    c.setStrokeColor(colors.HexColor("#D4D4D4"))
    c.setFillColor(colors.black)
    for exponent in range(0, int(math.ceil(x_log_max)) + 1):
        tick = 10**exponent
        if tick > x_max:
            continue
        x = sx(tick)
        c.line(x, bottom, x, bottom + plot_height)
        c.drawCentredString(x, bottom - 12, f"10^{exponent}")
    for exponent in range(int(math.floor(y_log_min)), 1):
        tick = 10**exponent
        if tick < y_min:
            continue
        y = sy(tick)
        c.line(left, y, left + plot_width, y)
        c.drawRightString(left - 7, y - 2.5, f"10^{exponent}")
    c.setStrokeColor(colors.black)
    c.rect(left, bottom, plot_width, plot_height, stroke=1, fill=0)
    return sx, sy


def create_history_ccdf(profiles: list[DatasetProfile], path: Path) -> None:
    width, height = 520.0, 335.0
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_header(c, "Distribution of user-history and session lengths", width, height)
    left, bottom, plot_width, plot_height = 76.0, 63.0, 330.0, 225.0
    x_max = max(float(p.entity_counts.max()) for p in profiles)
    y_min = max(1e-6, min(1.0 / p.entity_counts.size for p in profiles))
    sx, sy = draw_log_axes(c, left, bottom, plot_width, plot_height, x_max, y_min)
    draw_axis_labels(
        c,
        left,
        bottom,
        plot_width,
        plot_height,
        "Interactions in a user history or session (log scale)",
        "Share with at least this many interactions (log scale)",
    )

    for index, profile in enumerate(profiles):
        values, frequencies = np.unique(profile.entity_counts, return_counts=True)
        ccdf = np.cumsum(frequencies[::-1])[::-1] / profile.entity_counts.size
        values, ccdf = downsample_xy(values.astype(float), ccdf.astype(float))
        points = [(sx(float(x)), sy(max(float(y), y_min))) for x, y in zip(values, ccdf)]
        c.setStrokeColor(SERIES_COLORS[index])
        c.setLineWidth(1.5)
        if LINE_DASHES[index]:
            c.setDash(LINE_DASHES[index])
        else:
            c.setDash()
        path_object = c.beginPath()
        path_object.moveTo(*points[0])
        for point in points[1:]:
            path_object.lineTo(*point)
        c.drawPath(path_object, stroke=1, fill=0)
    c.setDash()
    draw_legend(c, profiles, 425, 275)
    c.setFont("Helvetica", 6.8)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(76, 18, "Source: processed RecBole interaction files used in this thesis.")
    c.save()


def create_lorenz(profiles: list[DatasetProfile], path: Path) -> None:
    width, height = 520.0, 335.0
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_header(c, "Concentration of interactions across catalogue items", width, height)
    left, bottom, plot_width, plot_height = 76.0, 63.0, 330.0, 225.0

    def sx(value: float) -> float:
        return left + value * plot_width

    def sy(value: float) -> float:
        return bottom + value * plot_height

    c.setFont("Helvetica", 7)
    c.setStrokeColor(colors.HexColor("#D4D4D4"))
    c.setFillColor(colors.black)
    for tick in np.linspace(0, 1, 6):
        x, y = sx(float(tick)), sy(float(tick))
        c.line(x, bottom, x, bottom + plot_height)
        c.line(left, y, left + plot_width, y)
        c.drawCentredString(x, bottom - 12, f"{tick:.1f}")
        c.drawRightString(left - 7, y - 2.5, f"{tick:.1f}")
    c.setStrokeColor(colors.HexColor("#777777"))
    c.setDash(3, 3)
    c.line(left, bottom, left + plot_width, bottom + plot_height)

    for index, profile in enumerate(profiles):
        ordered = np.sort(profile.item_counts.astype(float))
        cumulative = np.concatenate(([0.0], np.cumsum(ordered) / ordered.sum()))
        item_share = np.linspace(0.0, 1.0, cumulative.size)
        item_share, cumulative = downsample_xy(item_share, cumulative)
        points = [(sx(float(x)), sy(float(y))) for x, y in zip(item_share, cumulative)]
        c.setStrokeColor(SERIES_COLORS[index])
        c.setLineWidth(1.5)
        if LINE_DASHES[index]:
            c.setDash(LINE_DASHES[index])
        else:
            c.setDash()
        path_object = c.beginPath()
        path_object.moveTo(*points[0])
        for point in points[1:]:
            path_object.lineTo(*point)
        c.drawPath(path_object, stroke=1, fill=0)

    c.setDash()
    c.setStrokeColor(colors.black)
    c.rect(left, bottom, plot_width, plot_height, stroke=1, fill=0)
    draw_axis_labels(
        c,
        left,
        bottom,
        plot_width,
        plot_height,
        "Cumulative share of catalogue items, least popular first",
        "Cumulative share of interactions",
    )
    draw_legend(c, profiles, 425, 275)
    c.setFont("Helvetica", 6.8)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(76, 18, "The diagonal denotes an equal interaction count for every item.")
    c.save()


def bin_width_label(hours: float) -> str:
    if hours < 24:
        return f"{hours:.1f} hours/bin"
    if hours < 24 * 365:
        return f"{hours / 24:.1f} days/bin"
    return f"{hours / (24 * 365.25):.1f} years/bin"


def create_temporal_activity(profiles: list[DatasetProfile], path: Path) -> None:
    width, height = 520.0, 610.0
    c = canvas.Canvas(str(path), pagesize=(width, height))
    draw_header(c, "Interaction activity across each observation period", width, height)
    left, plot_width = 76.0, 400.0
    panel_height, gap = 78.0, 22.0
    top = height - 63.0
    bins = 24

    for index, profile in enumerate(profiles):
        bottom = top - panel_height - index * (panel_height + gap)
        hours = np.asarray(list(profile.hourly_counts.keys()), dtype=float)
        weights = np.asarray(list(profile.hourly_counts.values()), dtype=float)
        minimum = profile.timestamp_min / 3600.0
        maximum = profile.timestamp_max / 3600.0
        histogram, _ = np.histogram(hours, bins=bins, range=(minimum, maximum), weights=weights)
        shares = histogram / histogram.sum()
        y_max = max(float(shares.max()) * 1.08, 0.01)

        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.black)
        start = datetime.fromtimestamp(profile.timestamp_min, tz=timezone.utc).date().isoformat()
        end = datetime.fromtimestamp(profile.timestamp_max, tz=timezone.utc).date().isoformat()
        span_hours = (profile.timestamp_max - profile.timestamp_min) / bins / 3600.0
        c.drawString(left, bottom + panel_height + 6, profile.label)
        c.setFont("Helvetica", 6.8)
        c.drawRightString(
            left + plot_width,
            bottom + panel_height + 6,
            f"{start} to {end}; {bin_width_label(span_hours)}",
        )

        c.setStrokeColor(colors.HexColor("#D4D4D4"))
        for tick in (0.0, y_max / 2.0, y_max):
            y = bottom + tick / y_max * panel_height
            c.line(left, y, left + plot_width, y)
            c.setFillColor(colors.black)
            c.drawRightString(left - 7, y - 2.5, f"{tick * 100:.1f}%")

        bar_width = plot_width / bins
        c.setFillColor(SERIES_COLORS[index])
        for bin_index, value in enumerate(shares):
            x = left + bin_index * bar_width + 0.5
            bar_height = float(value) / y_max * panel_height
            c.rect(x, bottom, max(bar_width - 1.0, 0.5), bar_height, stroke=0, fill=1)
        c.setStrokeColor(colors.black)
        c.rect(left, bottom, plot_width, panel_height, stroke=1, fill=0)

    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, 23, "Equal-duration bins from the start to the end of each dataset")
    c.setFont("Helvetica", 6.8)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawString(76, 10, "Bars show each bin's share of all interactions in that dataset.")
    c.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "recbole",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "recbole_results" / "thesis_dataset_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        read_profile(args.data_root, dataset, label, entity_type)
        for dataset, label, entity_type in DATASETS
    ]
    statistics = pd.DataFrame(statistics_row(profile) for profile in profiles)
    statistics.to_csv(args.output_dir / "dataset_descriptive_statistics.csv", index=False)

    create_history_ccdf(profiles, args.output_dir / "history_length_ccdf.pdf")
    create_lorenz(profiles, args.output_dir / "item_popularity_lorenz.pdf")
    create_temporal_activity(profiles, args.output_dir / "temporal_activity.pdf")

    print(statistics.to_string(index=False))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
