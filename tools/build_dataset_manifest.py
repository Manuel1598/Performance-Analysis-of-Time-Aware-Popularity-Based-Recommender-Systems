"""Build a reproducibility manifest for the five prepared evaluation datasets."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "prepared_dataset_manifest.csv"
DATASETS = (
    ("Adressa sample", "data/recbole/adressa_recbole_sample/adressa_recbole_sample.inter"),
    (
        "Amazon Video Games (ratings >= 4)",
        "data/recbole/amazon_positive_recbole/amazon_positive_recbole.inter",
    ),
    ("Globo sample", "data/recbole/globo_recbole_sample/globo_recbole_sample.inter"),
    (
        "MovieLens 20M (ratings >= 4)",
        "data/recbole/movielens_positive_recbole/movielens_positive_recbole.inter",
    ),
    ("YOOCHOOSE sample", "data/recbole/yoochoose_recbole_sample/yoochoose_recbole_sample.inter"),
)


def describe(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            lines += block.count(b"\n")
    return path.stat().st_size, max(lines - 1, 0), digest.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "relative_path", "bytes", "interaction_rows", "sha256"))
        for name, relative in DATASETS:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            size, rows, digest = describe(path)
            writer.writerow((name, relative, size, rows, digest))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
