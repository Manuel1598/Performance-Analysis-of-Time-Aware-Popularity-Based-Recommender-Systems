# Audited VSKNN changes and consolidated best-model overview

## Purpose and selection rule

This file consolidates the best successful configuration of every model on every
available dataset. Selection uses the thesis primary metric **MRR@10**. If MRR@10
ties, the lower recorded runtime and then the original row order are used.

Legacy VS-KNN rows are deliberately excluded and replaced by the corrected,
performance-optimized, compact-tuned **audited VSKNN** rows.

## What changed in VSKNN and why

- Replaced unweighted cosine SKNN-like scoring with reference VSKNN position weighting.
- Added neighbor score decay based on the most recent shared click.
- Collapsed RecBole prefix-target augmentation to one training reference session per
  session ID, preventing duplicated longer sessions from being over-weighted.
- Kept the index restricted to `train_data.dataset`; validation/test targets are not
  inserted into the neighbor index.
- Removed the thesis-specific popularity correction from the upstream-faithful path.
- Standardized class/configuration names and retained a temporary compatibility alias.
- Replaced per-query global candidate sorting with a lazy merge of pre-sorted item
  session lists; quality is unchanged while runtime is substantially lower.
- Added algorithm, leakage, candidate-order, CLI, resume, and best-selection tests.
- Retuned only the corrected VSKNN; unchanged models and data splits were not rerun.

## Important comparability notes

- Ranking metrics can be compared within the recorded evaluation setup.
- Runtime is descriptive only when device/hardware differs. Audited VSKNN sample runs
  use CPU, while many stored legacy/framework results use CUDA.
- Top-N datasets and session datasets represent different recommendation scenarios and
  should not be ranked against each other.
- Session results currently use the three local sample datasets; Top-N results use the
  stored MovieLens and Amazon RecBole datasets.

## Best configuration of every model by dataset

### Session-based: `adressa_recbole_sample`

| Model | Hit@10 | NDCG@10 | MRR@10 | Runtime (s) |
| --- | --- | --- | --- | --- |
| GRU4Rec | 0.5220 | 0.2892 | 0.2181 | 26.18 |
| VSKNN | 0.4313 | 0.2312 | 0.1703 | 142.53 |
| VSTAN | 0.4204 | 0.2221 | 0.1619 | 2795.94 |
| RecentPop | 0.3815 | 0.1884 | 0.1296 | 6.48 |
| DecayPop | 0.3734 | 0.1841 | 0.1268 | 7.94 |
| MostPop | 0.3734 | 0.1841 | 0.1268 | 6.40 |

**Best by MRR@10:** GRU4Rec (0.2181).

### Session-based: `globo_recbole_sample`

| Model | Hit@10 | NDCG@10 | MRR@10 | Runtime (s) |
| --- | --- | --- | --- | --- |
| GRU4Rec | 0.3614 | 0.1988 | 0.1493 | 27.36 |
| VSTAN | 0.3644 | 0.1482 | 0.0831 | 287.73 |
| VSKNN | 0.3298 | 0.1341 | 0.0751 | 143.85 |
| DecayPop | 0.0626 | 0.0316 | 0.0224 | 8.57 |
| MostPop | 0.0626 | 0.0316 | 0.0224 | 6.93 |
| RecentPop | 0.0626 | 0.0316 | 0.0224 | 6.81 |

**Best by MRR@10:** GRU4Rec (0.1493).

### Session-based: `yoochoose_recbole_sample`

| Model | Hit@10 | NDCG@10 | MRR@10 | Runtime (s) |
| --- | --- | --- | --- | --- |
| VSKNN | 0.5377 | 0.3478 | 0.2880 | 190.56 |
| VSTAN | 0.5334 | 0.3384 | 0.2773 | 541.25 |
| GRU4Rec | 0.4818 | 0.3027 | 0.2469 | 32.13 |
| DecayPop | 0.0216 | 0.0115 | 0.0084 | 8.91 |
| MostPop | 0.0215 | 0.0112 | 0.0080 | 7.25 |
| RecentPop | 0.0212 | 0.0111 | 0.0080 | 7.26 |

**Best by MRR@10:** VSKNN (0.2880).

### Top-N: `amazon_recbole`

| Model | Hit@10 | NDCG@10 | MRR@10 | Runtime (s) |
| --- | --- | --- | --- | --- |
| BPR | 0.0754 | 0.0573 | 0.0519 | 12903.07 |
| DecayPop | 0.0260 | 0.0136 | 0.0100 | 1670.23 |
| MostPop | 0.0236 | 0.0124 | 0.0091 | 1662.26 |
| RecentPop | 0.0131 | 0.0055 | 0.0034 | 1730.85 |

**Best by MRR@10:** BPR (0.0519).

### Top-N: `movielens_recbole`

| Model | Hit@10 | NDCG@10 | MRR@10 | Runtime (s) |
| --- | --- | --- | --- | --- |
| BPR | 0.3546 | 0.0840 | 0.1546 | 5134.07 |
| DecayPop | 0.2510 | 0.0558 | 0.1065 | 423.75 |
| MostPop | 0.2479 | 0.0555 | 0.1062 | 284.62 |
| RecentPop | 0.1997 | 0.0395 | 0.0776 | 339.24 |

**Best by MRR@10:** BPR (0.1546).

## Source files

- `recbole_results/tuning_results/topn_full_tuning_results.csv`
- `recbole_results/tuning_results/session_full_tuning_results.csv`
- `recbole_results/vsknn_audited/compact_tuning_best_by_dataset.csv`
- generated detailed CSV: `recbole_results/summary/best_models_by_dataset.csv`
- generated dataset winners: `recbole_results/summary/best_overall_model_per_dataset.csv`

Regenerate this overview with:

```powershell
python -m src.recbole_framework.analysis.build_best_model_overview
```
