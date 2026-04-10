# Project Structure

This file documents the current code and folder structure of the thesis project.

The project is organized around a reproducible recommendation pipeline with separate modules for data preprocessing, models, evaluation, utilities, and analysis.

---

## 1. High-Level Structure

```text
data/
docs/
logs/
results/
src/
```

### `data/`

Stores raw, processed, and framework-specific dataset files.

### `docs/`

Contains documentation files such as dataset descriptions and project structure.

### `logs/`

Contains the project work log and development history.

### `results/`

Stores generated recommendation outputs, evaluation results, and analysis artifacts.

### `src/`

Contains the source code for preprocessing, models, evaluation, utilities, and analysis.

---

## 2. Source Code Structure

```text
src/
  analysis/
  datapipeline/
  evaluation/
  models/
  utils/
```

### `src/analysis/`

Contains scripts for comparing model results and generating analysis outputs such as tables and plots.

Typical outputs:

* consolidated comparison tables
* metric visualizations
* result summaries for thesis figures

### `src/datapipeline/`

Contains dataset-specific preprocessing and splitting scripts.

Typical responsibilities:

* load raw datasets
* convert data to a unified interaction format
* sort interactions chronologically
* create train/test splits
* prepare framework-specific files if needed

### `src/evaluation/`

Contains shared ranking metrics and evaluation logic.

Current responsibilities:

* compute ranking metrics such as HR, NDCG, and MRR
* evaluate recommendation files in a model-independent way
* support consistent comparison across methods

### `src/models/`

Contains recommender model implementations.

This module is divided into:

* shared model abstractions
* popularity-based recommenders
* RecBole-based model integrations

### `src/utils/`

Contains shared helper functions used across the project.

Current responsibilities:

* data loading and saving
* recommendation output formatting
* recommendation list construction
* helper functions for user history and ground-truth generation

---

## 3. Model Structure

```text
src/models/
  base.py
  popularity/
  recbole/
```

### `src/models/base.py`

Defines the shared recommender interface used across model implementations.

This provides a common structure for:

* fitting a model on training data
* generating recommendations for a user
* integrating new recommenders into the shared pipeline

---

### `src/models/popularity/`

Contains the popularity-based recommendation models and their corresponding runner scripts.

Current models:

* `mostpop.py`
* `recentpop.py`
* `decaypop.py`

Runner scripts:

* `run_mostpop.py`
* `run_recentpop.py`
* `run_decaypop.py`

Evaluation scripts:

* `evaluate_mostpop.py`
* `evaluate_recentpop.py`
* `evaluate_decaypop.py`

These models are implemented in an object-oriented structure and use shared pipeline components.

---

### `src/models/recbole/`

Contains RecBole-related dataset preparation scripts, wrappers, runner scripts, and evaluation scripts.

Current files:

* `prepare_recbole_movielens.py`
* `bpr_wrapper.py`
* `run_bpr.py`
* `evaluate_bpr.py`

This module integrates model-based baselines into the same experimental pipeline as the popularity-based models.

---

## 4. Data Flow

The project follows a modular pipeline:

```text
raw data
  ↓
preprocessing
  ↓
processed interaction data
  ↓
model training / recommendation generation
  ↓
recommendation output files
  ↓
evaluation
  ↓
metrics and analysis outputs
```

---

### Current Top-N Pipeline (MovieLens)

```text
raw MovieLens data
  ↓
preprocessing_movielens.py
  ↓
movielens_interactions.csv
  ↓
split.py
  ↓
movielens_train.csv + movielens_test.csv
  ↓
run_*.py
  ↓
results/*_recommendations.csv
  ↓
evaluate_*.py
  ↓
results/*_metrics.csv
  ↓
analysis scripts
```

---

## 5. Design Principles

### Reproducibility

All steps from preprocessing to evaluation are explicitly implemented and can be rerun.

### Modularity

Shared functionality is centralized and reused across models.

### Comparability

Different recommendation methods are evaluated under the same Top-N pipeline.

### Extensibility

The structure supports adding:

* additional Top-N models
* more RecBole-based baselines
* session-based recommendation models
* additional datasets

---

## 6. Current Status

At the current stage, the project includes:

* a working Top-N pipeline for MovieLens
* three popularity-based models:

  * MostPop
  * RecentPop
  * DecayPop
* one model-based baseline via RecBole:

  * BPR
* shared evaluation and utility modules
* result comparison and visualization support

The next step is the extension to additional datasets and session-based recommendation methods.
