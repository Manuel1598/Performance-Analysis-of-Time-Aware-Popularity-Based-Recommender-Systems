# Docker Server Runbook

This runbook explains how to start the full RecBole experiments on an external GPU server with Docker. It is written as a step-by-step checklist so the setup can be repeated without knowing the project internals.

## 1. What This Server Run Does

The Docker setup runs the thesis experiments on prepared RecBole datasets. The datasets are not copied into the Docker image. They must be mounted from the server filesystem into the container.

Default Top-N experiments:

- Datasets: `movielens_recbole`, `amazon_recbole`
- Models: `MostPop`, `RecentPop`, `DecayPop`
- Optional model: `BPR` with `--include-bpr`

Default session-based experiments:

- Datasets: `adressa_recbole`, `globo_recbole`, `yoochoose_recbole`
- Models: `MostPop`, `RecentPop`, `DecayPop`, `VS-KNN`, `VSTAN`, `GRU4Rec`

Important: the session run uses the full dataset names by default, not the local sample dataset names.

## 2. Requirements On The Server

The server needs:

- Linux server with NVIDIA GPU
- NVIDIA driver installed on the host
- Docker installed
- NVIDIA Container Toolkit installed
- enough disk space for datasets, Docker image, logs, and result CSV files
- this project repository available on the server

Check Docker:

```bash
docker --version
```

Check whether Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this command fails, the problem is usually the NVIDIA Container Toolkit or host driver setup, not this project.

## 3. Project Location On The Server

Clone or copy the repository to the server. In the examples below, the project is placed here:

```text
/server/projects/TimeAware-Popularity-Models
```

Go into the project root before building:

```bash
cd /server/projects/TimeAware-Popularity-Models
```

The project root is the folder that contains:

```text
Dockerfile
docker/requirements-server.txt
src/
docs/
```

## 4. Where The Data Must Be

Create one server directory that contains the prepared RecBole datasets:

```text
/server/data/timeaware/recbole/
```

Inside that folder, the structure must look like this:

```text
/server/data/timeaware/recbole/
  adressa_recbole/
    adressa_recbole.inter
  globo_recbole/
    globo_recbole.inter
  yoochoose_recbole/
    yoochoose_recbole.inter
  movielens_recbole/
    movielens_recbole.inter
  amazon_recbole/
    amazon_recbole.inter
```

The exact RecBole files can include additional generated files, but each dataset folder must at least contain the matching `.inter` file.

For example:

```text
adressa_recbole/adressa_recbole.inter
globo_recbole/globo_recbole.inter
yoochoose_recbole/yoochoose_recbole.inter
movielens_recbole/movielens_recbole.inter
amazon_recbole/amazon_recbole.inter
```

When Docker starts, this server folder is mounted to:

```text
/app/data/recbole/
```

That means the container will see:

```text
/app/data/recbole/adressa_recbole/
/app/data/recbole/globo_recbole/
/app/data/recbole/yoochoose_recbole/
/app/data/recbole/movielens_recbole/
/app/data/recbole/amazon_recbole/
```

## 5. Where Results Will Be Written

Create one result directory on the server:

```bash
mkdir -p /server/results/timeaware/recbole_results
```

This folder is mounted to:

```text
/app/recbole_results/
```

The default server run writes these files:

```text
/server/results/timeaware/recbole_results/tuning_results/server_full_topn_results.csv
/server/results/timeaware/recbole_results/experiment_logs/server_full_topn_log.csv
/server/results/timeaware/recbole_results/tuning_results/server_full_session_results.csv
/server/results/timeaware/recbole_results/experiment_logs/server_full_session_log.csv
```

The result files are separate from the local tuning files, so they do not overwrite the previous local CSVs.

## 6. Build The Docker Image

Build from the project root:

```bash
docker build -t timeaware-recbole .
```

The Dockerfile uses CUDA 12.8 and installs PyTorch from:

```text
https://download.pytorch.org/whl/cu128
```

If the server needs another CUDA wheel index, override it during build. Example for CUDA 12.4 wheels:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t timeaware-recbole .
```

## 7. First Smoke Test

Run this before starting the full experiment. It checks:

- Docker image works
- GPU is visible
- data mount path is correct
- result mount path is writable
- the runner can execute at least one Top-N and one session model

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --topn-datasets movielens_recbole \
  --session-datasets adressa_recbole \
  --topn-models MostPop \
  --session-models MostPop \
  --output-prefix smoke_test
```

After the smoke test, check whether these files exist:

```text
/server/results/timeaware/recbole_results/tuning_results/smoke_test_topn_results.csv
/server/results/timeaware/recbole_results/experiment_logs/smoke_test_topn_log.csv
/server/results/timeaware/recbole_results/tuning_results/smoke_test_session_results.csv
/server/results/timeaware/recbole_results/experiment_logs/smoke_test_session_log.csv
```

If the script prints a warning like:

```text
WARNING: dataset directory not found: /app/data/recbole/adressa_recbole
```

then the mounted data path is wrong or the dataset folder name does not match.

## 8. Start A Persistent Server Session

The full run can take a long time. Start it inside `tmux` so it keeps running after the SSH connection is closed.

Create a new `tmux` session:

```bash
tmux new -s recbole-full
```

Inside the `tmux` session, go to the project root:

```bash
cd /server/projects/TimeAware-Popularity-Models
```

Build the Docker image if it has not been built yet:

```bash
docker build -t timeaware-recbole .
```

Then start the full run from inside this `tmux` session.

Detach from `tmux` without stopping the run:

```text
Ctrl+B
d
```

This means: press `Ctrl+B`, release the keys, then press `d`.

Do not use `Ctrl+C` unless the run should really be stopped.

Reconnect later:

```bash
tmux attach -t recbole-full
```

List running `tmux` sessions:

```bash
tmux ls
```

If `tmux` is not installed, install it on the server or use `screen` as an alternative. For this project, `tmux` is the recommended option.

## 9. Run All Default Server Experiments

Start the full default run:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole
```

This works because the Dockerfile default command is:

```bash
python3 src/recbole_framework/tuning/run_server_full_experiments.py
```

The same command can also be written explicitly:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py
```

Important: for real server runs, execute this command inside the `tmux` session from the previous section.

## 10. Expected Number Of Runs

Default Top-N grid:

- `MostPop`: 1 run per dataset
- `RecentPop`: 8 runs per dataset
- `DecayPop`: 7 runs per dataset
- 16 runs per Top-N dataset
- 2 Top-N datasets
- Total Top-N default: 32 runs

Default session grid:

- `MostPop`: 1 run per dataset
- `RecentPop`: 5 runs per dataset
- `DecayPop`: 7 runs per dataset
- `VS-KNN`: 36 runs per dataset
- `VSTAN`: 216 runs per dataset
- `GRU4Rec`: 36 runs per dataset
- 301 runs per session dataset
- 3 session datasets
- Total session default: 903 runs

Total default without BPR:

```text
32 Top-N runs + 903 session runs = 935 runs
```

Optional Top-N BPR adds 9 runs per Top-N dataset, so 18 additional runs.

## 11. Run Only Top-N

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-session
```

Run Top-N including BPR:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-session \
  --include-bpr
```

## 12. Run Only Session Models

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-topn
```

## 13. Run Selected Models Or Datasets

Only run `VSTAN` on `globo_recbole`:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-topn \
  --session-datasets globo_recbole \
  --session-models VSTAN \
  --output-prefix server_vstan_globo
```

Only run the session popularity baselines:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-topn \
  --session-models MostPop RecentPop DecayPop \
  --output-prefix server_session_popularity
```

Use a smaller KNN grid:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-topn \
  --session-models VS-KNN VSTAN \
  --knn-k-values 200 \
  --knn-sample-sizes 500 2000 \
  --popularity-weights 0.0 1.0 \
  --vstan-position-decays 0.2 \
  --output-prefix server_knn_small
```

## 14. Resume After Stopping

The server runner is resumable. It reads the existing result CSVs and skips already successful `run_id`s.

If the run is stopped, start the same command again with the same `--output-prefix`. Already completed runs are skipped.

Example:

```bash
docker run --rm --gpus all \
  -v /server/data/timeaware:/app/data \
  -v /server/results/timeaware/recbole_results:/app/recbole_results \
  timeaware-recbole
```

The default prefix is:

```text
server_full
```

So the resume logic checks:

```text
server_full_topn_results.csv
server_full_session_results.csv
```

If you change `--output-prefix`, the run starts or resumes a different result set.

## 15. Useful Monitoring Commands

Watch GPU usage in another terminal:

```bash
nvidia-smi
```

Watch result files:

```bash
ls -lh /server/results/timeaware/recbole_results/tuning_results/
```

Show the last lines of a log:

```bash
tail -n 50 /server/results/timeaware/recbole_results/experiment_logs/server_full_session_log.csv
```

Count successful session runs:

```bash
grep -c ",success," /server/results/timeaware/recbole_results/tuning_results/server_full_session_results.csv
```

## 16. Common Problems

### Docker cannot access the GPU

Symptom:

```text
CUDA available: False
```

or Docker fails before the Python script starts.

Check:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If that fails, fix the server NVIDIA Docker setup first.

### Dataset directory not found

Symptom:

```text
WARNING: dataset directory not found: /app/data/recbole/globo_recbole
```

Check that the host path contains:

```text
/server/data/timeaware/recbole/globo_recbole/globo_recbole.inter
```

and that Docker is started with:

```bash
-v /server/data/timeaware:/app/data
```

### Results are not written

Check that the host result directory exists and is writable:

```bash
mkdir -p /server/results/timeaware/recbole_results
touch /server/results/timeaware/recbole_results/write_test.txt
rm /server/results/timeaware/recbole_results/write_test.txt
```

Then check that Docker is started with:

```bash
-v /server/results/timeaware/recbole_results:/app/recbole_results
```

### Runtime is too high

The expensive part is mostly `VSTAN` on full session datasets. To reduce runtime, start with a smaller grid:

```bash
--session-models VS-KNN VSTAN
--knn-k-values 200
--knn-sample-sizes 500 2000
--popularity-weights 0.0 1.0
--vstan-position-decays 0.2
```

Or run only one dataset first:

```bash
--session-datasets adressa_recbole
```

## 17. Final Output Files To Collect

After the full run, collect:

```text
/server/results/timeaware/recbole_results/tuning_results/server_full_topn_results.csv
/server/results/timeaware/recbole_results/experiment_logs/server_full_topn_log.csv
/server/results/timeaware/recbole_results/tuning_results/server_full_session_results.csv
/server/results/timeaware/recbole_results/experiment_logs/server_full_session_log.csv
```

These files can then be copied back locally and passed into the existing evaluation and reporting scripts.
