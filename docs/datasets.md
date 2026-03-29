# Dataset Documentation

This file documents all datasets used in the thesis project, including their source, local storage path, and intended purpose.

---

## 1. MovieLens 20M

### Purpose

Used as the primary dataset for reproducing the reference paper and for the initial comparison of popularity-based recommender models.

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

### Preprocessing

* reduced to implicit interactions (`user_id`, `item_id`, `timestamp`)
* interactions sorted chronologically per user
* users with fewer than 2 interactions were filtered (not needed for MovieLens 20M but included for consistency)

### Generated Files

* `data/processed/movielens_interactions.csv`
* `data/processed/movielens_train.csv`
* `data/processed/movielens_test.csv`

### Split Strategy

* chronological leave-one-out split
* last interaction per user → test set
* all previous interactions → training set

### Generated Recommendation Outputs

* `results/movielens_mostpop_recommendations.csv`
* `results/movielens_recentpop_recommendations.csv`
* `results/movielens_decaypop_recommendations.csv`

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


### Stored Metrics

The following ranking metrics are currently computed and stored:

* HR@5
* HR@10
* NDCG@5
* NDCG@10

### Purpose of Generated Outputs

The recommendation files and evaluation result files are used to compare the performance of popularity-based recommendation models on MovieLens.
They serve as the basis for comparing static popularity (MostPop) and time-aware popularity models (RecentPop and DecayPop).
This setup enables a controlled evaluation of how temporal information influences recommendation performance.



---

## 2. Amazon Reviews

### Purpose
Used as a second domain to test whether the findings generalize beyond movie recommendation.

### Source
Amazon Review Data / McAuley Lab

### Dataset Version / Time Reference
- downloaded dataset version: 2023
- used as the product recommendation domain in this thesis

### Local Path
`data/raw/amazon/`

### Downloaded Files
- `Electronics.json.gz`
- `Movies_and_TV.json.gz`

### Relevant Fields
- `reviewerID`
- `asin`
- `unixReviewTime`

### Role in Thesis
- product recommendation domain
- cross-domain comparison

---

## 3. MIND Dataset

### Purpose
Optional third domain for evaluating highly time-sensitive recommendation data.

### Source
Microsoft News Dataset (MIND)

### Dataset Version / Time Reference
- dataset version used in this thesis: 2020
- used as an optional news recommendation domain

### Local Path
`data/raw/mind/`

### Relevant Files
- `behaviors.tsv`
- `news.tsv`

### Relevant Fields
From `behaviors.tsv`:
- `user_id`
- `time`
- clicked news item

### Role in Thesis
- optional news recommendation domain
- evaluation under strong temporal dynamics

