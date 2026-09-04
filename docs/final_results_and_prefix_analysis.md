# Final results and prefix-length analysis

## Source policy

`tools.consolidate_final_results` reads the transferred PC2 worker directory and
the local `vstan_collapsed_v7` directory. It writes only to
`recbole_results/final_analysis`; the worker sources are never changed.

- All non-VSTAN results use `validation_first_v6` from PC2.
- All VSTAN validation and final results use
  `validation_first_v7_vstan_collapsed` from PC1.
- The old v1 final row, superseded VSTAN rows and BPR configurations with 512
  embedding dimensions are excluded explicitly.
- Protocol identifiers are not renamed. Each included final row retains the
  source path, selected validation identifier, host and checkpoint location.
- The merge checks the complete eligible grid, selection/tie-breaking, one
  seed-42 final per model/dataset, configurations and metric completeness.
- Required totals remain 197 validation rows and 26 final rows.

From the project root, using the experiment environment:

```powershell
.\.venv-vsknn\Scripts\python.exe -m tools.consolidate_final_results
.\.venv-vsknn\Scripts\python.exe -m tools.evaluate_session_prefix_groups
.\.venv-vsknn\Scripts\python.exe -m tools.build_final_results_tables
.\.venv-vsknn\Scripts\python.exe -m unittest tests.test_final_analysis
```

The prefix command supports `--models` and `--datasets` for a restricted replay.
It defaults to four CPU threads and one prediction process. For VS-KNN and
VSTAN, `--workers` can enable independent batch processes, each with one tensor
thread. Batch results are collected in the original order. These settings are for the supplementary
analysis only and must not be used to reinterpret the original runtime records.

The parallel Yoochoose VSTAN attempt failed the aggregate check: NDCG@5 was
0.31597241 against the stored 0.3159. Its groups were not published. Replaying
the same checkpoint in one process produced 0.31594504 and passed all six
checks. The tolerance was not relaxed. The precise internal source of the
parallel difference was not isolated; serial execution is now the default.
The other 17 published combinations passed their original checks. Each audit
records its actual worker count. The evaluator used for these results is
retained under `recbole_results/final_analysis/provenance`; the current runner
differs only in its safer default worker count.

## Prefix analysis

The replay uses the user's own transferred/generated checkpoints. PyTorch's
full checkpoint loading can execute pickle code, so do not substitute an
untrusted downloaded checkpoint. The saved configuration is checked against
the final row before data loading. Only the local data path and CPU placement
are changed; the selected hyperparameters and split remain fixed.

No `fit` call is made. GRU4Rec loads learned weights. Non-parametric models
reconstruct their counts or historical index from the same training data.
The original Top-10 masking and torch top-k semantics are retained.

Groups describe **available input clicks**, not total eventual session length:
1, 2, 3, and 4+. The original maximum retained prefix is 20. An empty group has
missing metrics, not a zero score. The existing split supplies no one-click
test inputs; very short sequences may contribute training examples only.

For every model/dataset, the replay must match the six original aggregate
ranking metrics to within 0.000051, the tolerance for four-decimal CSV values.
It must also reproduce the query count. Group-weighted metrics reconstruct the
unrounded replay to an absolute tolerance of 1e-12. Publication additionally
requires identical query fingerprints and group counts across the six models
on each dataset.

Outputs include compact compressed per-query prefix lengths and target ranks,
per-model group metrics and an audit JSON. A target rank of 0 means the target
was not in the Top-10. It is not a rank above the first item. The per-query
records deliberately omit original user and item identifiers.

## Interpretation

PC1 VSTAN runtimes are not controlled comparisons with PC2 models. Primary
results contain one seed and no seed-level confidence interval. The prefix
analysis is a descriptive subgroup analysis, not a truncation intervention:
different groups also contain different test queries and targets. It must not
be used to select a new model configuration after seeing the test results.

The table builder updates only the generated final tables in the thesis repo.
Chapter text is edited separately. A Git commit or push is not part of these
commands.
