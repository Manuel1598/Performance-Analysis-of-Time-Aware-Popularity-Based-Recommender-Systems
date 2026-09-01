# Validation-first v6 result consolidation

This pipeline combines worker outputs without relying on their base file names.
Two files named `validation_trials.csv` remain distinct because the source
manifest records their complete paths and labels.

## Declared sources

The source list is stored in
`config/validation_first_v6_result_sources.json`:

- main validation results, including MovieLens;
- the copied Amazon PC 2 worker file;
- the copied session PC 2 worker file;
- the stopped 512-dimensional Amazon PC 1 attempt retained for the audit trail;
- the central final-test result file.

Replace copied worker files only after the process writing them has stopped.
Do not merge directly into a file that still has an active writer.

## Conflict policy

Rows are matched by `run_id` for validation and by `final_test_id` for final
tests.

1. An exact duplicate is kept once.
2. A successful validation retry replaces an older failed row with the same ID.
3. Two successful validation rows may differ only in validation runtime. The
   source listed first in the manifest is retained and the decision is audited.
4. Different successful metrics or configurations for one ID stop the pipeline.
5. Final-test duplicates are stricter: runtime is reported scientifically, so
   any non-exact successful duplicate stops the pipeline.

No source row is silently edited. Every choice is written to
`merge_decisions.csv`, and source hashes are written to `source_inventory.csv`.

## Provisional run

Use this while tuning or final testing is still running:

```powershell
.venv-vsknn\Scripts\python.exe -m tools.run_validation_first_v6_reporting --allow-incomplete
```

The command writes an incomplete but internally checked snapshot below
`recbole_results/validation_first_v6_consolidated`. It expects 197 successful,
selection-eligible validation trials, 26 required seed-42 final test runs, and
26 final model/dataset summary groups. Existing BPR trials above 256 dimensions
are retained in the audit but do not count as required or eligible. Missing
required IDs are listed explicitly.

## Strict final run

After all worker files have been copied and every final test has completed:

```powershell
.venv-vsknn\Scripts\python.exe -m tools.run_validation_first_v6_reporting
```

Without `--allow-incomplete`, missing sources, missing required runs, unknown
IDs, or conflicting successful duplicates stop the command. Declared seed-43
robustness rows are allowed but not required. A successful strict run has
`state: complete` in `audit_report.json`.

## Outputs

The consolidated directory contains:

- `validation_trials_merged.csv`: one canonical row per validation `run_id`,
  including the explicit selection-eligibility fields;
- `validation_successes.csv`: successful, selection-eligible validation rows;
- `validation_excluded_by_budget.csv`: preserved BPR rows above the
  256-dimensional resource limit and their exclusion reason;
- `selected_validation_configs.csv`: one eligible configuration per model and
  dataset using validation MRR@10, runtime, and `run_id` as tie breakers;
- `final_test_results_merged.csv`: one canonical row per final test and seed;
- `final_test_summary.csv`: primary seed-42 metrics and runtime; optional
  seed-43 values and differences are stored in separate columns;
- `completion_status.csv` and `missing_runs.csv`: completeness checks;
- `source_inventory.csv` and `merge_decisions.csv`: audit trail;
- `audit_report.json`: machine-readable protocol status;
- `final_topn_results.tex` and `final_session_results.tex`: generated LaTeX
  result tables;
- `workbook_payload.json`: source for the review workbook.

The workbook payload can be made display-safe and passed to the artifact-tool
builder. The generated workbook separates source inventory, merged validation,
selected configurations, final raw results, final summaries, decisions, and
missing runs into distinct sheets.

## Tests

```powershell
.venv-vsknn\Scripts\python.exe -m unittest \
  tests.test_consolidate_validation_first_v6 \
  tests.test_run_validation_first_v6_reporting \
  tests.test_merge_validation_first_results -v
```

The tests cover equal base names in different directories, success-over-failure
replacement, runtime-only validation duplicates, conflicting successful
metrics, expected protocol counts, winner tie breaking, the 256-dimensional
selection limit, and separation of primary and optional seed results.
