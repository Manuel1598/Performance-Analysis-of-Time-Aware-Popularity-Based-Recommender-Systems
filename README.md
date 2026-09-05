# Temporal Effects in Recommender Systems

**A Comparative Evaluation Across Top-N and Session-Based Recommendation**

Master thesis project by Manuel Weilguni, Alpen-Adria-Universitaet Klagenfurt,
2026.

## Project status

The experimental code and the primary evaluation are complete. The repository
contains the RecBole integrations, validation-first experiment runner, audited
result consolidation, final analysis, and tests used for the thesis.

The final result set contains:

- 197 successful and selection-eligible validation trials;
- 26 final model--dataset evaluations on the held-out test partitions;
- one primary final seed (`42`) per model--dataset pair;
- 8 Top-N results and 18 session results;
- a verified supplementary analysis by available session-prefix length.

The 26 final configurations were selected by validation MRR@10 before their
test scores were evaluated. Test results were not used for model selection.

## Experimental scope

### Top-N recommendation

Datasets:

- `amazon_recbole` (Amazon Video Games)
- `movielens_recbole` (MovieLens 20M)

Models:

- `MostPop`: global popularity
- `RecentPop`: popularity in a recent time window
- `DecayPop`: exponentially time-decayed popularity
- `BPR`: personalised Bayesian pairwise ranking

### Session-based recommendation

Prepared 500,000-event samples:

- `adressa_recbole_sample`
- `globo_recbole_sample`
- `yoochoose_recbole_sample`

Models:

- `MostPop`
- `RecentPop`
- `DecayPop`
- `VS-KNN`: session-neighbourhood recommendation
- `VSTAN`: sequence- and time-aware session-neighbourhood recommendation
- `GRU4Rec`: recurrent next-item recommendation

Only these prepared session samples contribute to the thesis results. Full
session datasets and earlier prototype runs are not part of the final tables.

## Final evaluation protocol

The primary runner is `tools.run_validation_first_experiments`.

- Interactions are ordered chronologically (`TO`).
- RecBole applies an 80/10/10 ratio split within each persistent user or
  retained sequence identity.
- Sequential datasets are first converted into prefix--target examples.
- Hyperparameters are selected separately for every model--dataset pair by
  validation MRR@10.
- Exact validation ties are resolved by lower validation runtime and then by
  the stable run identifier.
- The selected configuration is refitted and evaluated once with seed `42`.
- Every primary final run uses CPU.
- BPR configurations above 256 embedding dimensions are retained in the audit
  trail but excluded from selection because of the declared resource limit.
- Corrected VSTAN results reconstruct one historical training reference per
  original sequence. They replace the superseded VSTAN rows from protocol v6.

VS-KNN and VSTAN do not insert every RecBole training prefix as a separate
historical neighbour. Their adapters collapse the augmented rows into one
reference per original training sequence. This prevents longer sequences from
receiving extra neighbour weight only because they generate more prefixes.

The corrected VSTAN runs were recorded on different CPU hardware from the
other final runs. Their quality values belong to the final comparison, but
their runtimes are excluded from direct cross-model speed comparisons and the
quality--runtime Pareto frontier.

## Environment and data

Python 3.11 is the supported project environment. A compact CPU environment
can be created with:

```powershell
py -3.11 -m venv .venv-vsknn
.\.venv-vsknn\Scripts\python.exe -m pip install -r requirements-session-worker.txt
```

Prepared data is expected below `data/recbole/` in RecBole `.inter` format:

```text
data/recbole/<dataset>/<dataset>.inter
```

Raw and prepared datasets are intentionally not committed. Reproduction
requires the same processed `.inter` files, dependency versions, configurations
and code revision, not only the same random seed.

## Running the validation-first experiments

Run or resume all validation trials:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.run_validation_first_experiments `
  --phase tune `
  --scenario both `
  --device cpu
```

Run the final test phase after all required validation trials are complete:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.run_validation_first_experiments `
  --phase test `
  --scenario both
```

The runner also accepts `--datasets`, `--models`, and `--output-dir`. A separate
output directory can isolate a worker process. Successful stable identifiers
are skipped when a stopped run is resumed.

By default, the runner writes incrementally to:

- `recbole_results/validation_first/validation_trials.csv`
- `recbole_results/validation_first/final_test_results.csv`
- `recbole_results/validation_first/checkpoints/`

Seed `43` is available only as an optional robustness check for BPR and
GRU4Rec through `--include-optional-robustness-seed`. It is not required for
completion and does not replace the primary seed-42 result.

## Consolidating the final results

The immutable worker sources expected by the final consolidation are:

- `recbole_results/validation_first_workers/final_pc2/`
- `recbole_results/validation_first_workers/vstan_collapsed_v7/`

Their directory names preserve the execution provenance. Run:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.consolidate_final_results
```

The command verifies the complete eligible grids, the validation winners, the
frozen final configurations, seed 42, checkpoint uniqueness, result metrics,
and expected totals. It writes only to `recbole_results/final_analysis/` and
does not modify the source files.

Important final files are:

- `recbole_results/final_analysis/audit.json`
- `recbole_results/final_analysis/validation_trials.csv`
- `recbole_results/final_analysis/selected_validation.csv`
- `recbole_results/final_analysis/final_results.csv`
- `recbole_results/final_analysis/runtime_efficiency.csv`

The source policy keeps protocol-v6 rows for unchanged models and replaces all
VSTAN validation and final rows with the corrected protocol-v7 results. Legacy
final rows, superseded VSTAN rows, and ineligible 512-dimensional BPR trials are
excluded explicitly rather than silently rewritten.

## Session-prefix analysis

The supplementary analysis groups the existing session test queries by one,
two, three, and at least four available input clicks:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.evaluate_session_prefix_groups
```

This command performs inference only. It reloads the frozen final checkpoints
and never calls `fit`. A group is published only if all six aggregate ranking
metrics reproduce the original rounded final result, query counts match, and
all models use identical query fingerprints and group sizes.

The final split contains no one-click test queries, so that group is recorded
as empty rather than assigned a score of zero. The groups contain different
queries and targets; they are descriptive subgroups, not a controlled test of
what happens when another click is added to the same query.

Outputs are stored below:

```text
recbole_results/final_analysis/prefix_groups/
```

## Metrics

Primary selection metric:

- `MRR@10`

Additional ranking metrics:

- `Hit@5`, `Hit@10`
- `NDCG@5`, `NDCG@10`
- `MRR@5`

Catalogue and recommendation-distribution metrics:

- `coverage@10`
- `unique_recommended_items@10`
- `avg_recommendation_popularity@10`
- `recommendation_frequency_gini@10`

Runtime fields separate total execution, ranking evaluation, and the additional
catalogue-metric pass. Recorded totals are descriptive single-run measurements,
not repeated hardware benchmarks or per-request serving latency.

## Generating final thesis tables

After consolidation and prefix replay, generate the LaTeX tables with:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.build_final_results_tables
```

The table builder reads only the audited final analysis. It never creates a
thesis result from a validation score. The optional table export expects the
separate thesis checkout at `overleaf-thesis-project/`; that nested repository
is intentionally not tracked by this code repository.

## Tests

Run the complete test suite with the Python standard library:

```powershell
.\.venv-vsknn\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains 52 tests covering, among other things:

- stable validation identifiers and resumable result writing;
- conflict-safe consolidation of distributed result files;
- grid completeness, resource exclusions, and winner selection;
- final-result and prefix-group reconciliation;
- VS-KNN scoring and candidate retrieval;
- collapse of RecBole prefixes for VS-KNN and VSTAN;
- protection against validation/test rows entering the neighbour index.

## Repository structure

```text
config/                       declared result sources and protocol settings
data/recbole/                 local prepared datasets (not committed)
docs/                         detailed audit and reproduction notes
logs/                         retained execution evidence and work log
recbole_results/final_analysis/
                              audited compact final results
src/recbole_framework/        models, dataset preparation, runners and analysis
tests/                        unit and protocol tests
tools/                        final experiment, consolidation and replay tools
```

Earlier prototype and exploratory code is retained for traceability. The
validation-first runner and `recbole_results/final_analysis/` are authoritative
for the thesis results.

## Detailed documentation

- `docs/final_results_and_prefix_analysis.md`
- `docs/validation_first_v6_reporting_pipeline.md`
- `docs/recbole_vsknn_audit.md`
- `docs/recbole_vsknn_upstream_checklist.md`
- `docs/reproducibility.md`

## Reproducibility boundaries

The primary results use one seed and one temporal split. Small score differences
therefore do not establish statistical significance or stability across time
periods. The session experiments use prepared samples, and VSTAN runtime was
measured on different hardware. These limits are part of the documented result
interpretation, not hidden post-processing choices.
