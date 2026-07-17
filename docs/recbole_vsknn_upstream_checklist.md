# VSKNN upstream checklist for RecBole

## Scope decision

The first contribution contains VSKNN only. VSTAN remains a separate future
proposal because it has not yet completed the same paper/reference audit,
leakage review, optimization validation, and upstream-focused test preparation.

## Phase 1: proposal issue

- [x] Explain why VSKNN is useful as a non-parametric session baseline.
- [x] Cite the paper and authors' reference implementation.
- [x] Describe the proposed `SequentialRecommender` integration.
- [x] State that the neighbor index is built only from `train_data.dataset`.
- [x] Explain reconstruction of one training session from augmented rows.
- [x] List supported similarity, sampling, and weighting choices.
- [x] Provide deterministic correctness and leakage test coverage.
- [x] Report multi-dataset RecBole 1.2.1 sample results without cherry-picking.
- [x] Report the compact tuning winners and selection metric.
- [x] Ask maintainers how a non-parametric model should integrate with Trainer.
- [ ] Publish `docs/recbole_vsknn_issue_draft.md` in the official issue tracker.
- [ ] Record the official issue URL in the worklog.

## Phase 2: pull request after maintainer feedback

Only these upstream-facing components should be transferred into a RecBole fork:

1. `recbole/model/sequential_recommender/vsknn.py`
   - `VSKNN` model class;
   - reference-faithful weighting/similarity/scoring helpers;
   - training-session reconstruction;
   - deterministic recent candidate selection;
   - `predict` and `full_sort_predict`.
2. `recbole/properties/model/VSKNN.yaml`
   - `neighbor_size: 100`;
   - `sample_size: 1000`;
   - `sampling: recent`;
   - `similarity: vec`;
   - `session_weighting: div`;
   - `score_weighting: div`.
3. RecBole model registration/import changes required by the maintainers.
4. `docs/source/user_guide/model/sequential/vsknn.rst` and the sequential
   model index entry.
5. Deterministic unit/integration tests adapted to RecBole's own test layout.

The exact training hook and whether helper functions remain in the model module
or a separate internal module will follow the maintainer decision in the issue.

## Explicitly excluded from upstream

- thesis tuning and experiment runners;
- result CSV, Excel, plots, and worklog files;
- dataset preparation scripts;
- the thesis-specific popularity correction;
- temporary legacy configuration keys (`vsknn_k`, `vsknn_sample_size`);
- the temporary `VSKNNRecBole` compatibility alias;
- VSTAN.

## Local evidence ready for the PR

- algorithm audit: `docs/recbole_vsknn_audit.md`;
- issue text: `docs/recbole_vsknn_issue_draft.md`;
- adapter: `src/recbole_framework/custom_models/session/vsknn_recbole.py`;
- framework-independent core: `src/recbole_framework/custom_models/session/vsknn_core.py`;
- correctness tests: `tests/test_vsknn_core.py`;
- candidate-order/model tests: `tests/test_vsknn_model.py`;
- runner and tuning tests: `tests/test_vsknn_runner.py`,
  `tests/test_vsknn_tuning.py`;
- reproducible compact results:
  `recbole_results/vsknn_audited/compact_tuning_results.csv`.

## Acceptance checks before opening the PR

- [ ] Rebase the RecBole fork on the maintainer-requested target branch.
- [ ] Apply the agreed non-parametric lifecycle design.
- [ ] Remove thesis compatibility aliases and legacy parameter names.
- [ ] Run RecBole's complete model test suite.
- [ ] Run formatting/lint checks used by the official repository.
- [ ] Re-run at least one deterministic smoke test from the clean fork.
- [ ] Ensure documentation, YAML defaults, and code parameter names match.
- [ ] Link the proposal issue from the PR description.
