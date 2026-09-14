"""Exploratory paired tests for the final session recommendation results.

The prefix replay files contain the rank of the held-out target for every test
query in a stable order.  This script compares the dataset-level MRR@10 winner
with each other model using a two-sided Wilcoxon signed-rank test and applies a
Holm correction across all comparisons.  Several queries can originate from
one sequence, so the output is explicitly a query-level sensitivity analysis,
not an independent-user significance test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "recbole_results" / "final_analysis" / "prefix_groups"
OUTPUT = ROOT / "recbole_results" / "final_analysis" / "session_pairwise_significance.csv"

WINNERS = {
    "adressa_recbole_sample": "GRU4Rec",
    "globo_recbole_sample": "GRU4Rec",
    "yoochoose_recbole_sample": "VS-KNN",
}
MODELS = ("MostPop", "RecentPop", "DecayPop", "VS-KNN", "VSTAN", "GRU4Rec")


def reciprocal_rank_at_10(ranks: pd.Series) -> np.ndarray:
    values = ranks.to_numpy(dtype=float)
    result = np.zeros_like(values)
    np.divide(1.0, values, out=result, where=values > 0)
    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for position, index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def load_queries(dataset: str, model: str) -> pd.DataFrame:
    path = INPUT / f"{dataset}__{model}_queries.csv.gz"
    frame = pd.read_csv(path)
    required = ["prefix_length", "target_rank_at_10", "group"]
    if list(frame.columns) != required:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    return frame


def main() -> None:
    rows: list[dict[str, object]] = []
    for dataset, winner in WINNERS.items():
        reference = load_queries(dataset, winner)
        reference_rr = reciprocal_rank_at_10(reference["target_rank_at_10"])
        for competitor in MODELS:
            if competitor == winner:
                continue
            comparison = load_queries(dataset, competitor)
            if not reference[["prefix_length", "group"]].equals(
                comparison[["prefix_length", "group"]]
            ):
                raise ValueError(
                    f"Query order differs for {dataset}: {winner} vs {competitor}"
                )
            comparison_rr = reciprocal_rank_at_10(comparison["target_rank_at_10"])
            differences = reference_rr - comparison_rr
            statistic, p_value = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="approx",
            )
            rows.append(
                {
                    "dataset": dataset,
                    "winner": winner,
                    "competitor": competitor,
                    "query_count": len(differences),
                    "winner_mrr_at_10": reference_rr.mean(),
                    "competitor_mrr_at_10": comparison_rr.mean(),
                    "mean_difference": differences.mean(),
                    "winner_better_queries": int((differences > 0).sum()),
                    "competitor_better_queries": int((differences < 0).sum()),
                    "tied_queries": int((differences == 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )

    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result["p_value"].tolist())
    result["significant_at_0_05"] = result["holm_adjusted_p"] < 0.05
    result.to_csv(OUTPUT, index=False, float_format="%.10g")
    print(result.to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
