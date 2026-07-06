# Reproducibility Guide

This guide explains how to reproduce the project experiments and result tables in a scientifically traceable way.

The goal is that another person can start from the repository, use the same prepared data format, rerun the experiments, and regenerate the structured result artifacts.

## 1. Reproducibility Principle

The project follows a staged reproducibility workflow:

1. prepare datasets in RecBole `.inter` format
2. run the model tuning experiments
3. write raw result CSVs and experiment logs
4. generate structured evaluation tables, plots, and markdown reports
5. report the exact code version, dataset version, result files, and evaluation metric

The final RecBole-based pipeline is the authoritative pipeline for the thesis results. The older prototype code remains in the repository for traceability, but it is not the main source for final results.

## 2. Code Version

For a reproducible run, record the Git branch and commit hash before starting experiments:

```bash
git branch --show-current
git rev-parse HEAD
```

For the Docker/server preparation, the intended branch is:

```text
docker-server-prepare
```

The exact commit hash should be written down together with the generated result files. This makes it possible to connect every result table to the code that produced it.

## 3. Environment Options

There are two supported ways to reproduce results.

### Option A: Docker Server Run

This is the recommended option for long full-dataset runs on a GPU server.

Use:

```text
Dockerfile
docker/requirements-server.txt
docs/server_docker_run.md
src/recbole_framework/tuning/run_server_full_experiments.py
```

The Docker setup installs the experiment environment inside the container and avoids relying on a local Windows or IDE Python environment.

### Option B: Local Python Environment

This is useful for smaller runs, debugging, and report generation.

Install the local requirements in a valid Python environment:

```bash
python -m pip install -r requirements.txt
```

The local `requirements.txt` is broader than the Docker requirements because it also contains local notebook and development tooling.

## 4. Dataset Requirements

All final experiments use RecBole-formatted datasets under:

```text
data/recbole/
```

Expected Top-N datasets:

```text
data/recbole/movielens_recbole/movielens_recbole.inter
data/recbole/amazon_recbole/amazon_recbole.inter
```

Expected session-based datasets:

```text
data/recbole/adressa_recbole/adressa_recbole.inter
data/recbole/globo_recbole/globo_recbole.inter
data/recbole/yoochoose_recbole/yoochoose_recbole.inter
```

Some local development and presentation results use sample dataset names:

```text
adressa_recbole_sample
globo_recbole_sample
yoochoose_recbole_sample
```

For full server reproduction, use the full dataset names unless the research question explicitly concerns the sample datasets.

## 5. Dataset Provenance To Record

For scientific reproducibility, record:

- original raw dataset source
- download date or dataset release/version
- preprocessing script used
- final `.inter` file path
- number of interactions
- number of users or sessions
- number of items
- timestamp range

The structured evaluator computes dataset characteristics from the `.inter` files and writes them to:

```text
recbole_results/tuning_results/analysis_results/structured_report/dataset_summary.csv
```

## 6. Experimental Settings

The final tuning scripts use RecBole evaluation settings with:

```text
metrics = Hit, NDCG, MRR
topk = 5, 10
valid_metric = MRR@10
seed = 42
reproducibility = True
```

The main quality metric for interpretation is:

```text
MRR@10
```

Supporting metrics:

```text
Hit@5
Hit@10
NDCG@5
NDCG@10
MRR@5
```

Additional bias and efficiency metrics:

```text
coverage@10
avg_recommendation_popularity@10
runtime_seconds
train_runtime_seconds
eval_runtime_seconds
extra_metrics_runtime_seconds
mrr@10_per_minute
```

Runtime can vary between machines, but the recorded runtime values are still important for comparing models within the same hardware environment.

## 7. Full Docker Reproduction

For the complete server run, follow:

```text
docs/server_docker_run.md
```

Minimal command sequence on the server:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
git checkout docker-server-prepare
docker build -t timeaware-recbole .
```

Start a persistent session:

```bash
tmux new -s recbole-full
```

Run the full experiment inside `tmux`:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole
```

Detach without stopping:

```text
Ctrl+B
d
```

Reconnect:

```bash
tmux attach -t recbole-full
```

## 8. Server Output Files

The server runner writes separate output files with the `server_full` prefix:

```text
recbole_results/tuning_results/server_full_topn_results.csv
recbole_results/experiment_logs/server_full_topn_log.csv
recbole_results/tuning_results/server_full_session_results.csv
recbole_results/experiment_logs/server_full_session_log.csv
```

These files are written continuously while the run progresses.

The run is resumable. If the same command is restarted with the same `--output-prefix`, successful existing `run_id`s are skipped.

## 9. Local Full-Tuning Output Files

The local full-tuning scripts use these default filenames:

```text
recbole_results/tuning_results/topn_full_tuning_results.csv
recbole_results/experiment_logs/topn_full_tuning_experiment_log.csv
recbole_results/tuning_results/session_full_tuning_results.csv
recbole_results/experiment_logs/session_full_tuning_experiment_log.csv
```

These are the default input files for the structured evaluator with:

```bash
python src/recbole_framework/analysis/evaluate_recbole_results.py --scope full
```

## 10. Converting Server Results For The Existing Evaluator

The current structured evaluator expects the local full-tuning filenames when `--scope full` is used.

After copying server results back, either keep an archive of the original server files and copy them to the expected names:

```bash
cp recbole_results/tuning_results/server_full_topn_results.csv \
   recbole_results/tuning_results/topn_full_tuning_results.csv

cp recbole_results/experiment_logs/server_full_topn_log.csv \
   recbole_results/experiment_logs/topn_full_tuning_experiment_log.csv

cp recbole_results/tuning_results/server_full_session_results.csv \
   recbole_results/tuning_results/session_full_tuning_results.csv

cp recbole_results/experiment_logs/server_full_session_log.csv \
   recbole_results/experiment_logs/session_full_tuning_experiment_log.csv
```

or create a separate analysis folder and document exactly which input files were used.

Important: never overwrite the only copy of a result file. Keep the original raw server CSVs as immutable experiment artifacts.

## 11. Generate Structured Evaluation

Run:

```bash
python src/recbole_framework/analysis/evaluate_recbole_results.py --scope full
```

The structured report is written to:

```text
recbole_results/tuning_results/analysis_results/structured_report/
```

Main generated files:

```text
dataset_summary.csv
model_summary.csv
tuning_summary.csv
best_overall.csv
best_per_model.csv
comparative_summary.csv
runtime_summary.csv
efficiency_summary.csv
popularity_weighting_summary.csv
recbole_structured_evaluation.md
plots/
```

For thesis interpretation, the most important files are:

```text
comparative_summary.csv
best_per_model.csv
runtime_summary.csv
popularity_weighting_summary.csv
recbole_structured_evaluation.md
```

## 12. Minimum Artifact Package For Reproduction

To allow another person to verify or reproduce the result interpretation, keep this package:

```text
Git branch and commit hash
Dockerfile
docker/requirements-server.txt
docs/server_docker_run.md
docs/reproducibility.md
prepared RecBole .inter dataset files
raw result CSVs
experiment log CSVs
structured_report/
```

Recommended directory layout for an archived run:

```text
reproduction_package/
  code_commit.txt
  data_manifest.md
  server_command.txt
  recbole_results/
    tuning_results/
    experiment_logs/
    tuning_results/analysis_results/structured_report/
```

## 13. What Must Be Reported In The Thesis

For each final experiment group, report:

- dataset name
- model name
- tuning configuration
- primary metric `MRR@10`
- supporting metrics `Hit@10` and `NDCG@10`
- runtime
- hardware environment
- code commit
- whether sample or full dataset was used

For time-aware models, report the relevant time hyperparameter:

- `RecentPop`: `window_days` or `recent_fraction`
- `DecayPop`: `decay_lambda` or `decay_half_life_days`
- `VS-KNN`: `vsknn_sample_size`, `vsknn_popularity_weight`
- `VSTAN`: `vstan_sample_size`, `vstan_position_decay`, `vstan_idf_weighting`, `vstan_popularity_weight`

## 14. Expected Limitations

Exact runtime values may differ between machines because they depend on:

- GPU model
- CPU model
- disk speed
- Docker version
- CUDA/PyTorch version
- concurrent server load

Ranking metrics should be reproducible when:

- the same prepared `.inter` files are used
- the same code commit is used
- the same model configuration grid is used
- RecBole reproducibility settings remain enabled

Small floating-point differences can still occur across hardware and library versions, especially for neural models such as `GRU4Rec`.

## 15. Quick Checklist

Before running:

- record Git commit
- verify dataset folder names
- verify each `.inter` file exists
- build Docker image
- run smoke test
- start full run in `tmux`

During running:

- monitor `nvidia-smi`
- monitor experiment logs
- do not use `Ctrl+C` unless stopping is intended

After running:

- archive raw result CSVs
- archive experiment logs
- generate structured report
- record final commit and hardware
- compare final tables against thesis claims
