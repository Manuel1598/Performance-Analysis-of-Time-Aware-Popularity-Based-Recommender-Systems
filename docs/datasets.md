# Dataset Documentation

This file documents all datasets used in the thesis project, including their source, local storage path, intended role in the thesis, preprocessing assumptions, and generated outputs where applicable.

---

# 1. Top-N Recommendation Datasets

## 1.1 MovieLens 20M

### Purpose

Used as the primary Top-N dataset for reproducing the reference study and for the initial comparison of popularity-based and model-based recommender systems.

### Source

GroupLens Research

### Dataset Version / Time Reference

* interaction period ends in 2015
* dataset publication/update reference: 2016
* this thesis uses the downloaded version available at the start of the project

### Local Path

`data/raw/movielens/`

### Expected Main File

`ratings.csv`

### Relevant Fields

* `userId`
* `movieId`
* `timestamp`

### Role in Thesis

* reproduction of the reference study
* first benchmark for MostPop, RecentPop, and DecayPop
* controlled Top-N recommendation setting for initial experiments
* first comparison between popularity-based models and a model-based collaborative filtering baseline via RecBole
* initial RecBole integration for reproducible comparison under the same evaluation pipeline

### Preprocessing

* reduced to implicit interactions (`user_id`, `item_id`, `timestamp`)
* interactions sorted chronologically per user
* users with fewer than 2 interactions were filtered (mainly for consistency with the general pipeline)

### Generated Files

* `data/processed/movielens_interactions.csv`
* `data/processed/movielens_train.csv`
* `data/processed/movielens_test.csv`
* `data/recbole/movielens_recbole/movielens_recbole.inter`

### Split Strategy

* chronological leave-one-out split
* last interaction per user → test set
* all previous interactions → training set

### Generated Recommendation Outputs

* `results/movielens_mostpop_recommendations.csv`
* `results/movielens_recentpop_recommendations.csv`
* `results/movielens_decaypop_recommendations.csv`
* `results/movielens_bpr_recommendations.csv`

### Recommendation Output Format

Each recommendation file contains one row per recommended item:

* `user_id`
* `rank`
* `item_id`

Each user receives a ranked top-k recommendation list (currently k = 10).

### Evaluation Result Files

* `results/movielens_mostpop_metrics.csv`
* `results/movielens_recentpop_metrics.csv`
* `results/movielens_decaypop_metrics.csv`
* `results/movielens_bpr_metrics.csv`

### Stored Metrics

The following ranking metrics are currently computed and stored:

* HR@5
* HR@10
* NDCG@5
* NDCG@10
* MRR@5
* MRR@10

### Purpose of Generated Outputs

The recommendation files and evaluation result files are used to compare the performance of popularity-based recommendation models on MovieLens.
They serve as the basis for comparing static popularity (MostPop), time-aware popularity models (RecentPop and DecayPop), and a first model-based baseline (BPR via RecBole).

This setup enables a controlled and reproducible evaluation of how temporal information influences recommendation performance in a Top-N setting.

### Analysis Outputs

* `results/analysis_results/movielens_popularity_model_comparison.csv`
* `results/analysis_results/movielens_hr10_comparison.png`
* `results/analysis_results/movielens_ndcg10_comparison.png`

### Purpose of Analysis Outputs

These files summarize and visualize the comparative performance of MostPop, RecentPop, DecayPop, and BPR on MovieLens.
They are used for result interpretation and for preparing thesis figures and tables.

---

## 1.2 Amazon Reviews

### Purpose

Used as a second Top-N recommendation domain to test whether findings generalize beyond movie recommendation.

### Source

Amazon Review Data / McAuley Lab

### Dataset Version / Time Reference

* downloaded dataset version: 2023
* used as the product recommendation domain in this thesis

### Local Path

`data/raw/amazon/`

### Downloaded Files

* `Electronics.json.gz`
* `Movies_and_TV.json.gz`

### Relevant Fields

* `reviewerID`
* `asin`
* `unixReviewTime`

### Intended Unified Fields After Preprocessing

* `user_id`
* `item_id`
* `timestamp`

### Role in Thesis

* second Top-N recommendation domain
* cross-domain comparison for popularity-based and model-based methods
* validation of whether observed effects also appear outside the movie domain

### Planned Processing

* convert raw review data into implicit interaction format
* reduce to `user_id`, `item_id`, `timestamp`
* apply chronological preprocessing
* generate train/test splits compatible with the Top-N evaluation pipeline

---

# 2. Session-Based Recommendation Datasets

## 2.1 Yoochoose

### Purpose

Used as a session-based e-commerce dataset for evaluating recommendation methods under short-term user intent and session dynamics.

### Source

RecSys Challenge 2015 / Yoochoose dataset

### Local Path

`data/raw/yoochoose/`

### Expected Main Files

* `yoochoose-clicks.dat`
* `yoochoose-buys.dat`

### Relevant Fields

From click data:
* `session_id`
* `timestamp`
* `item_id`
* `category`

From buy data:
* `session_id`
* `timestamp`
* `item_id`

### Role in Thesis

* main session-based e-commerce dataset
* evaluation of recommendation methods under session-based interaction patterns
* comparison of popularity-based and sequential/model-based methods in a session setting

### Planned Processing

* construct sessions from click sequences
* define recommendation targets at session level
* align data format with session-based evaluation pipeline

---

## 2.2 Globo

### Purpose

Used as a session-based news recommendation dataset with strong temporal dynamics.

### Source

Globo.com news interaction dataset

### Local Path

`data/raw/globo/`

### Expected Main Content

News portal interaction logs with session-based user behavior

### Relevant Fields

Expected core fields after preprocessing:
* `session_id`
* `item_id`
* `timestamp`

### Role in Thesis

* session-based news recommendation domain
* evaluation of methods on time-sensitive news consumption
* comparison between e-commerce-like and news-like session behavior

### Planned Processing

* convert interactions into session-based format
* preserve chronological order within sessions
* prepare data for session-based recommendation experiments

---

## 2.3 Adressa

### Purpose

Used as an additional news recommendation dataset to study temporal dynamics and news recommendation behavior.

### Source

Adressa dataset for news recommendation

### Local Path

`data/raw/adressa/`

### Expected Main Files

Depending on the downloaded subset/version, interaction logs and article metadata

### Relevant Fields

Expected core fields after preprocessing:
* `user_id` or session-level identifier
* `item_id`
* `timestamp`

### Role in Thesis

* additional news recommendation domain
* comparison with Globo in a temporally sensitive recommendation setting
* optional complementary dataset for session-aware and time-aware news evaluation

### Planned Processing

* convert raw interaction logs into a session-oriented or temporally ordered interaction format
* preserve strong time information for later evaluation
* adapt the dataset to the session-based recommendation pipeline where feasible

---

# 3. General Notes

### Current Experimental Focus

The project currently has two main evaluation tracks:

* **Top-N recommendation**
  * MovieLens 20M
  * Amazon Reviews

* **Session-based recommendation**
  * Yoochoose
  * Globo
  * Adressa

### Common Design Principle

All datasets are intended to be integrated into a reproducible experimental pipeline with:

* explicit preprocessing steps
* documented train/test splitting strategy
* consistent recommendation output format
* shared ranking-based evaluation where applicable

### Current Implementation Status

At the moment, the Top-N pipeline is already implemented and tested on MovieLens.
The session-based datasets are part of the planned next stages of the thesis.