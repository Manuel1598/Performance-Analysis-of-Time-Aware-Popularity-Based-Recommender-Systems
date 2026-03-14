# Time-Aware Popularity Models in Recommender Systems

Master Thesis Project  
Author: Manuel Weilguni  
University: AAU 
Year: 2026

---

# Project Overview

This repository contains the implementation and experiments for the master thesis:

**"Time-Aware Popularity Models and Fair Evaluation in Recommender Systems – A Cross-Domain Analysis"**

The goal of this work is to investigate whether classical popularity baselines used in recommender systems produce misleading results when temporal information is ignored.

The project starts by reproducing a reference study on the MovieLens dataset and later extends the analysis to additional domains such as Amazon product reviews and news recommendation datasets.

---

# Research Motivation

Popularity-based recommenders are often used as baseline models in recommender system research.

The most common baseline is **MostPop**, which ranks items purely by the total number of interactions in the training dataset.

However, this approach ignores the **temporal dynamics of popularity**. Items can be popular only during certain time periods, and ignoring this can lead to unrealistic evaluations.

This project investigates time-aware popularity models such as:

- MostPop (classic popularity baseline)
- RecentPop (recent popularity)
- DecayPop (time-decayed popularity)

The goal is to evaluate whether considering time leads to more realistic and fair baseline performance.

---

# Research Questions

The central research question of this thesis is:

> How strongly does incorporating temporal information influence the quality and fairness of popularity-based recommender baselines across different domains?

Sub-questions include:

- Does time-aware popularity improve recommendation accuracy?
- Do these effects generalize across different domains (movies, products, news)?
- How does user activity influence the effectiveness of popularity-based recommendations?

---

# Datasets

The following datasets are used in this project.

## MovieLens

MovieLens is used as the initial dataset to reproduce the reference paper results.

Dataset characteristics:

- Movie ratings dataset
- Includes timestamps of interactions
- Widely used benchmark for recommender systems

Used for:

- reproduction of baseline models
- controlled experimental setup

---

## Amazon Reviews

Amazon product review data is used as a second domain to evaluate whether the results generalize beyond movie recommendations.

Characteristics:

- user-product interactions
- timestamps available
- multiple product categories

Example categories used:

- Electronics
- Movies & TV

---

## MIND Dataset

The Microsoft News Dataset (MIND) is optionally used as a third domain.

Characteristics:

- news recommendation dataset
- strong temporal dynamics
- user click logs

This dataset allows evaluation in a **highly time-sensitive domain**.

---

# Methodology

The experiments follow a reproducible pipeline:

1. Data preprocessing
2. Creation of interaction dataset (user, item, timestamp)
3. Chronological dataset split
4. Implementation of popularity-based recommendation models
5. Offline evaluation using ranking metrics

---

# Implemented Models

The following baseline models are implemented.

### MostPop

Ranks items based on the total number of interactions in the training dataset.

This is the most widely used baseline in recommender systems.

---

### RecentPop

Ranks items based on their popularity within a recent time window.

Only interactions within a defined time interval before the recommendation time are considered.

---

### DecayPop

Ranks items using a time-decayed popularity score.

Recent interactions receive higher weights using an exponential decay function.

---

# Evaluation Metrics

The models are evaluated using standard ranking metrics:

- Hit Rate @ 5
- Hit Rate @ 10
- NDCG @ 5
- NDCG @ 10

These metrics measure whether the recommended items contain the ground-truth interaction.

---

# Reproducibility

All steps of the project are documented in: logs



