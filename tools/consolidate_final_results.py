"""Build the thesis result set without rewriting historical protocol identifiers.

v6 is authoritative for unchanged models; only VSTAN is replaced by v7.
All inputs are immutable. Selection is checked against the current eligible grid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import run_validation_first_experiments as exp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "recbole_results" / "final_analysis"
SOURCES = {
    "pc2": ROOT / "recbole_results/validation_first_workers/final_pc2",
    "pc1_vstan": ROOT / "recbole_results/validation_first_workers/vstan_collapsed_v7",
}


def source_for(model: str) -> tuple[str, str]:
    return ("pc1_vstan", "validation_first_v7_vstan_collapsed") if model == "VSTAN" else ("pc2", "validation_first_v6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def select_one(candidates: pd.DataFrame) -> pd.Series:
    if candidates.empty or not candidates.status.eq("success").all():
        raise ValueError("Missing or unsuccessful eligible validation candidates")
    if candidates.run_id.duplicated().any():
        raise ValueError("Duplicate validation identifiers")
    return candidates.sort_values(
        ["valid_mrr@10", "runtime_seconds", "run_id"],
        ascending=[False, True, True], kind="stable",
    ).iloc[0]


def validate_final(row: pd.Series, winner: pd.Series) -> None:
    if row.status != "success" or row.evaluated_split != "test" or float(row.seed) != 42:
        raise ValueError("Final row is not a successful seed-42 test")
    if row.selected_validation_run_id != winner.run_id:
        raise ValueError("Final row does not reference the validation winner")
    if json.loads(row.config_json) != json.loads(winner.config_json):
        raise ValueError("Final configuration differs from selected validation")
    if not np.isclose(row["selection_valid_mrr@10"], winner["valid_mrr@10"], rtol=0, atol=1e-10):
        raise ValueError("Selection metric mismatch")
    for metric in exp.FINAL_AGGREGATE_METRICS:
        value = float(row[metric])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric: {metric}")
        if metric != "avg_recommendation_popularity@10" and value > 1:
            raise ValueError(f"Metric outside [0,1]: {metric}")


def main() -> None:
    frames, inventory = {}, []
    for label, folder in SOURCES.items():
        frames[label] = {}
        for name in ("validation_trials.csv", "final_test_results.csv"):
            path = folder / name
            frames[label][name] = pd.read_csv(path)
            inventory.append({"source": label, "file": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    validations, finals, selections = [], [], []
    for scenario, datasets, models, grid in (
        ("topn", exp.TOPN_DATASETS, exp.TOPN_MODELS, exp.topn_grid),
        ("session", exp.SESSION_DATASETS, exp.SESSION_MODELS, exp.session_grid),
    ):
        for dataset in datasets:
            for model in models:
                label, protocol = source_for(model)
                v = frames[label]["validation_trials.csv"]
                v = v[(v.protocol_version == protocol) & (v.scenario == scenario) & (v.dataset == dataset) & (v.model == model) & (v.evaluated_split == "validation")].copy()
                expected = {exp.serialise_config(x) for x in grid(model)}
                v["canonical_config"] = v.config_json.map(lambda x: exp.serialise_config(json.loads(x)))
                v = v[v.canonical_config.isin(expected)].copy()
                if set(v.canonical_config) != expected or len(v) != len(expected):
                    raise ValueError(f"Incomplete or duplicate grid: {dataset}/{model}")
                winner = select_one(v)
                f = frames[label]["final_test_results.csv"]
                f = f[(f.protocol_version == protocol) & (f.scenario == scenario) & (f.dataset == dataset) & (f.model == model) & (f.seed == 42)]
                if len(f) != 1:
                    raise ValueError(f"Expected exactly one final: {dataset}/{model}")
                row = f.iloc[0].copy()
                validate_final(row, winner)
                checkpoint_paths = list((SOURCES[label] / "checkpoints" / scenario / dataset / model / "seed_42").glob("*.pth"))
                if len(checkpoint_paths) != 1:
                    raise ValueError(f"Ambiguous checkpoint: {dataset}/{model}")
                row["checkpoint_path"] = str(checkpoint_paths[0].relative_to(ROOT))
                row["source_file"] = str((SOURCES[label] / "final_test_results.csv").relative_to(ROOT))
                row["execution_host"] = label
                row["result_policy"] = "v6_unchanged_models_plus_v7_vstan"
                winner = winner.copy()
                winner["source_file"] = str((SOURCES[label] / "validation_trials.csv").relative_to(ROOT))
                validations.append(v.assign(source_file=winner["source_file"]))
                selections.append(winner)
                finals.append(row)
    merged = pd.DataFrame(finals).sort_values(["scenario", "dataset", "model"])
    validation = pd.concat(validations, ignore_index=True)
    assert len(merged) == 26 and len(validation) == 197
    assert not merged.duplicated(["scenario", "dataset", "model", "seed"]).any()
    if merged.groupby("dataset").recommendation_count.nunique().max() != 1:
        raise ValueError("Recommendation counts differ within dataset")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(merged, OUTPUT / "final_results.csv")
    atomic_csv(validation.drop(columns="canonical_config"), OUTPUT / "validation_trials.csv")
    atomic_csv(pd.DataFrame(selections).drop(columns="canonical_config"), OUTPUT / "selected_validation.csv")
    audit = {
        "policy": "Keep v6 unchanged-model rows; replace all VSTAN validation/final rows with v7. Preserve original identifiers.",
        "validation_successes": len(validation), "final_successes": len(merged),
        "sources": inventory,
        "excluded": ["v1 legacy final row", "v6 VSTAN rows superseded by v7", "BPR embedding_size=512 outside eligible grid"],
        "runtime_caveat": "VSTAN ran on PC1; other final results ran on PC2. Do not pool host-specific runtime comparisons.",
    }
    temporary_audit = OUTPUT / "audit.json.tmp"
    temporary_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    temporary_audit.replace(OUTPUT / "audit.json")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
