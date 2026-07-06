# Performance Analysis of Time-Aware Popularity-Based Recommender Systems

**Master thesis project**  
Author: Manuel Weilguni  
University: AAU  
Year: 2026

## Overview

This repository contains the implementation, tuning runs, and analysis pipeline
for the thesis project:

**Performance Analysis of Time-Aware Popularity-Based Recommender Systems**

The project studies whether simple popularity-based recommender systems become
stronger and more informative baselines when temporal information is included.
The main focus is on:

- ranking quality
- temporal popularity dynamics
- popularity bias
- runtime and computational efficiency
- differences between Top-N and session-based recommendation

The final experimental system is based on **RecBole**. Earlier prototype code is
kept in the repository for traceability, but the main results are produced with
the RecBole framework.

## Current Project Status

The project has moved beyond the prototype phase. The current state includes:

- RecBole-native custom Top-N models:
  - `MostPop`
  - `RecentPop`
  - `DecayPop`
- RecBole-compatible session models:
  - `VS-KNN`
  - `VSTAN`
  - `GRU4Rec`
- RecBole Top-N baseline:
  - `BPR`
- prepared RecBole datasets in `.inter` format
- full tuning result files for Top-N and session experiments
- structured analysis reports, comparison tables, runtime summaries, and plots

The current full-tuning result files are:

- `recbole_results/tuning_results/session_full_tuning_results.csv`
- `recbole_results/tuning_results/topn_full_tuning_results.csv`
- `recbole_results/experiment_logs/session_full_tuning_experiment_log.csv`
- `recbole_results/experiment_logs/topn_full_tuning_experiment_log.csv`

The structured evaluation report is generated here:

- `recbole_results/tuning_results/analysis_results/structured_report/recbole_structured_evaluation.md`

## Research Questions

The central research question is:

> How does incorporating temporal information affect the performance and
> behavior of popularity-based recommender systems?

Sub-questions:

- Do time-aware popularity models improve ranking performance compared with
  standard popularity baselines?
- How do popularity-based models compare with model-based recommender systems?
- How much popularity bias is visible in the recommendations?
- Do results differ across domains and datasets?
- How do Top-N and session-based recommendation results differ?
- Are quality improvements computationally efficient when runtime is included?

## Datasets

The project uses RecBole-formatted datasets under `data/recbole/`.

### Top-N Recommendation

- `movielens_recbole`
- `amazon_recbole`

### Session-Based Recommendation

- `yoochoose_recbole_sample`
- `globo_recbole_sample`
- `adressa_recbole_sample`

The structured analysis computes dataset characteristics directly from the
`.inter` files, including:

- number of interactions
- number of users or sessions
- number of items
- average interactions per user/session
- average interactions per item
- interaction matrix density
- timestamp range

## Models

### Top-N Models

Custom RecBole models:

- `MostPop`: global popularity baseline
- `RecentPop`: popularity within a recent time window
- `DecayPop`: popularity with time-decayed interaction weights

Baseline:

- `BPR`: Bayesian Personalized Ranking

### Session-Based Models

Custom session models:

- `VS-KNN`: session-neighborhood baseline
- `VSTAN`: time-aware session-neighborhood model

Baseline:

- `GRU4Rec`: neural session recommendation model

## Evaluation Metrics

The main evaluation metric for model quality is:

- `MRR@10`

`MRR@10` is used as the primary metric because it rewards models that rank the
first relevant item very high. This is especially important for session-based
recommendation, where the next useful item should appear near the top.

Supporting ranking metrics:

- `Hit@5`
- `Hit@10`
- `NDCG@5`
- `NDCG@10`
- `MRR@5`

Popularity-bias and recommendation-diversity metrics:

- `coverage@10`
- `avg_recommendation_popularity@10`

Runtime and efficiency metrics:

- `runtime_seconds`
- `train_runtime_seconds`
- `eval_runtime_seconds`
- `extra_metrics_runtime_seconds`
- `runtime_minutes`
- `mrr@10_per_minute`
- `ndcg@10_per_minute`
- `hit@10_per_minute`
- `runtime_relative_to_dataset_fastest`
- `quality_runtime_pareto_efficient`

The final analysis should not rely on ranking metrics alone. For this project,
the most important combined view is:

1. `MRR@10` for primary quality
2. `NDCG@10` for ranked relevance quality
3. `Hit@10` for intuitive top-k success
4. `coverage@10` for recommendation breadth
5. `avg_recommendation_popularity@10` for popularity bias
6. `runtime_seconds` and `mrr@10_per_minute` for efficiency

## Project Structure

```text
data/
  raw/                         raw downloaded datasets
  processed/                   prototype-stage processed data
  recbole/                     RecBole .inter datasets

docs/                          additional project documentation
logs/                          work log
notebooks/                     exploratory notebooks
recbole_results/               RecBole tuning logs, result CSVs, reports, plots
results_prototype/             prototype-stage outputs

src/
  prototype/                   initial standalone implementation
  recbole_framework/
    analysis/                  comparison and structured evaluation scripts
    custom_models/             custom RecBole and session model implementations
    datasets/                  dataset preparation scripts
    measurement/               extra metrics and experiment logging
    runners/                   single-model runner scripts
    tuning/                    tuning and full-experiment scripts
```

## Important Scripts

### Dataset Preparation

Top-N:

- `src/recbole_framework/datasets/topn/prepare_recbole_movielens.py`
- `src/recbole_framework/datasets/topn/prepare_recbole_amazon.py`

Session:

- `src/recbole_framework/datasets/session/prepare_yoochoose_recbole.py`
- `src/recbole_framework/datasets/session/prepare_yoochoose_recbole_sample.py`
- `src/recbole_framework/datasets/session/prepare_globo_recbole.py`
- `src/recbole_framework/datasets/session/prepare_adressa_recbole.py`

### Full Tuning

- `src/recbole_framework/tuning/tune_session_models_full.py`
- `src/recbole_framework/tuning/evaluate_session_models_final.py`
- `src/recbole_framework/tuning/tune_topn_models_full.py`
- `src/recbole_framework/tuning/run_all_full_tuning.py`

### Analysis

- `src/recbole_framework/analysis/analyze_session_tuning_results.py`
- `src/recbole_framework/analysis/analyze_topn_tuning_results.py`
- `src/recbole_framework/analysis/compare_recbole_session_models.py`
- `src/recbole_framework/analysis/compare_recbole_topn_models.py`
- `src/recbole_framework/analysis/evaluate_recbole_results.py`

The main analysis entry point is:

```powershell
python src\recbole_framework\analysis\evaluate_recbole_results.py --scope full
```

If the local virtual environment launcher is broken, use a valid Python 3.12
interpreter and make sure the project environment packages are available.

## Structured Result Analysis

The structured evaluator reads the full tuning results and creates:

- `dataset_summary.csv`
- `model_summary.csv`
- `tuning_summary.csv`
- `best_overall.csv`
- `best_per_model.csv`
- `comparative_summary.csv`
- `runtime_summary.csv`
- `efficiency_summary.csv`
- plots under `structured_report/plots/`
- `recbole_structured_evaluation.md`

The `comparative_summary.csv` file is the most useful compact table for thesis
interpretation because it combines quality, bias-related metrics, runtime, and
efficiency.

## Current Full-Tuning Scope

The current structured report is based on the full tuning files and contains:

- 356 cleaned result rows
- 355 successful result rows
- 5 datasets
- 7 evaluated models

Evaluated datasets:

- `adressa_recbole_sample`
- `amazon_recbole`
- `globo_recbole_sample`
- `movielens_recbole`
- `yoochoose_recbole_sample`

Evaluated models:

- `BPR`
- `DecayPop`
- `GRU4Rec`
- `MostPop`
- `RecentPop`
- `VS-KNN`
- `VSTAN`

## Reproducibility Notes

The project is designed around reproducible experiment stages:

1. prepare raw datasets into RecBole `.inter` files
2. run RecBole model tuning
3. store experiment logs and result CSVs
4. generate structured reports and plots from result CSVs

The generated result directories are experiment artifacts and may be ignored by
Git depending on local settings. The code required to reproduce the analysis is
kept under `src/recbole_framework/`.

For a complete step-by-step reproduction workflow, including Docker server runs,
dataset placement, output files, result archiving, and structured evaluation,
see:

- `docs/reproducibility.md`
- `docs/server_docker_run.md`

## Summary

This repository now contains both the historical prototype pipeline and the
current RecBole-based experimental system. The current project focus is no
longer basic model integration, but systematic comparison of Top-N and
session-based recommender models with respect to:

- ranking performance
- time-aware popularity effects
- popularity bias
- dataset differences
- runtime and efficiency

