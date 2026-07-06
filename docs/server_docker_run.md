# Docker Server Runs

This document describes how to run the full RecBole experiments on an external GPU server.

## Scope

The Docker setup is intended for full server-side experiment runs with mounted data and result directories.

Default server run:

- Top-N datasets:
  - `movielens_recbole`
  - `amazon_recbole`
- Top-N models:
  - `MostPop`
  - `RecentPop`
  - `DecayPop`
- Session datasets:
  - `adressa_recbole`
  - `globo_recbole`
  - `yoochoose_recbole`
- Session models:
  - `MostPop`
  - `RecentPop`
  - `DecayPop`
  - `VS-KNN`
  - `VSTAN`
  - `GRU4Rec`

`BPR` for Top-N can be enabled with `--include-bpr`.

## Build

Build the image from the project root:

``` bash
docker build -t timeaware-recbole .
```

The Dockerfile installs PyTorch from the CUDA 12.8 wheel index by default:

``` text
https://download.pytorch.org/whl/cu128
```

If the server requires a different PyTorch CUDA wheel index, override it:

``` bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t timeaware-recbole .
```

## Data and Results

Do not bake datasets into the Docker image.

Mount the prepared RecBole data directory and a result directory into the container:

``` bash
docker run --rm --gpus all \
  -v /server/path/data:/app/data \
  -v /server/path/recbole_results:/app/recbole_results \
  timeaware-recbole
```

The mounted data directory must contain:

``` text
data/recbole/adressa_recbole/
data/recbole/globo_recbole/
data/recbole/yoochoose_recbole/
data/recbole/movielens_recbole/
data/recbole/amazon_recbole/
```

## Outputs

The default server script writes separate files so local experiment outputs are not overwritten:

``` text
recbole_results/tuning_results/server_full_topn_results.csv
recbole_results/experiment_logs/server_full_topn_log.csv
recbole_results/tuning_results/server_full_session_results.csv
recbole_results/experiment_logs/server_full_session_log.csv
```

The run is resumable. Existing successful `run_id`s are skipped when the same command is restarted.

## Run Everything

``` bash
docker run --rm --gpus all \
  -v /server/path/data:/app/data \
  -v /server/path/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py
```

## Run Only Top-N

``` bash
docker run --rm --gpus all \
  -v /server/path/data:/app/data \
  -v /server/path/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-session
```

## Run Only Session Models

``` bash
docker run --rm --gpus all \
  -v /server/path/data:/app/data \
  -v /server/path/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --skip-topn
```

## Smaller Smoke Test

Use this first to verify Docker, GPU visibility, mounted data paths, and result writing:

``` bash
docker run --rm --gpus all \
  -v /server/path/data:/app/data \
  -v /server/path/recbole_results:/app/recbole_results \
  timeaware-recbole \
  python3 src/recbole_framework/tuning/run_server_full_experiments.py \
  --topn-datasets movielens_recbole \
  --session-datasets adressa_recbole \
  --topn-models MostPop \
  --session-models MostPop \
  --output-prefix smoke_test
```

## Notes

- `VS-KNN` and `VSTAN` can be very expensive on full session datasets.
- The default KNN sample-size grid is `500 1000 2000 5000`.
- The default popularity-weight grid for `VS-KNN` and `VSTAN` is `0.0 0.5 1.0`.
- The default session popularity baselines use the refined parameters:
  - `RecentPop recent_fraction = 0.01 0.05 0.10 0.25 0.50`
  - `DecayPop decay_half_life_days = 0.25 0.5 1 3 7 14 30`
- If runtime is too high, reduce the grid with CLI flags such as:

``` bash
--knn-sample-sizes 500 2000
--popularity-weights 0.0
--gru-epochs 20
```
