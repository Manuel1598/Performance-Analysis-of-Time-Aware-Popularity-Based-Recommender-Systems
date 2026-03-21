# Performance Analysis of Time-Aware Popularity-Based Recommender Systems

**Master Thesis Project**
Author: Manuel Weilguni
University: AAU
Year: 2026

---

# Project Overview

This repository contains the implementation and experiments for the master thesis:

**"Performance Analysis of Time-Aware Popularity-Based Recommender Systems"**

The goal of this work is to analyze the performance of popularity-based recommendation methods, with a particular focus on **temporal dynamics**, **trendiness**, and their impact on recommendation quality.

The project starts with the implementation of classical popularity baselines and extends the analysis to time-aware variants and comparisons with more advanced recommendation models.

---

# Research Motivation

Popularity-based recommender systems are widely used as baseline methods in research due to their simplicity and surprisingly strong performance.

The standard baseline, **MostPop**, ranks items based on their total number of interactions in the training data. However, this approach ignores that:

* popularity changes over time
* items may only be popular within specific time periods
* evaluation without temporal awareness can be misleading

Recent work has shown that incorporating time into popularity definitions can significantly improve performance and lead to more realistic evaluation setups .

This thesis revisits popularity-based recommendation and investigates whether **time-aware models provide better and more realistic baselines**.

---

# Research Objectives

The main objectives of this thesis are:

* Implement and analyze popularity-based recommendation models:

  * MostPop
  * RecentPop
  * DecayPop

* Investigate **temporal dynamics and trendiness** in recommendation systems

* Compare popularity-based approaches with **model-based recommender systems**:

  * BPR
  * NeuMF
  * GRU4Rec

* Analyze the impact of **popularity bias and fairness**

* Evaluate whether findings generalize across **multiple domains**

---

# Research Questions

> How does incorporating temporal information affect the performance and behavior of popularity-based recommender systems?

Sub-questions:

* Do time-aware popularity models improve recommendation accuracy?
* How do popularity-based methods compare to model-based approaches?
* How does popularity bias influence recommendation outcomes?
* Do results generalize across different datasets and domains?

---

# Datasets

The project uses multiple datasets to ensure cross-domain validity.

## Non-session-based datasets

* **MovieLens** (primary dataset for controlled experiments)
* **Amazon Reviews** (product recommendation domain)

## Session-based datasets

* **Globo dataset** (news recommendation)
* **Yoochoose dataset** (e-commerce sessions)
* Optional: **Adressa dataset**

---

# Methodology

The experiments follow a reproducible pipeline:

1. Data preprocessing (user_id, item_id, timestamp)
2. Chronological data splitting (leave-one-out)
3. Implementation of popularity-based models
4. Generation of recommendation outputs
5. Evaluation using ranking metrics
6. Comparison with advanced models using RecBole

The setup ensures **time-aware evaluation** and avoids **future data leakage**, following best practices from prior work .

---

# Implemented Models

## Popularity-Based Models

### MostPop

Ranks items based on global popularity in the training data.

### RecentPop

Ranks items based on interactions within a recent time window.

### DecayPop

Uses a time-decay function to weight recent interactions more strongly.

---

## Model-Based Baselines (via RecBole)

* BPR (Bayesian Personalized Ranking)
* NeuMF (Neural Matrix Factorization)
* GRU4Rec (session-based recommendation)

---

# Evaluation Metrics

The models are evaluated using standard ranking metrics:

* NDCG@k
* MRR@k

Additionally, the project analyzes:

* popularity distribution
* potential bias and fairness aspects

---

# Reproducibility

All steps of the project are documented and reproducible:

* preprocessing pipeline
* dataset splits
* model outputs
* evaluation results

See:

* `logs/` (work log)
* `data/processed/`
* `results/`

---

# Current Status

The following components are already implemented:

* MovieLens preprocessing pipeline
* Chronological leave-one-out split
* MostPop baseline
* Evaluation pipeline (HR, NDCG)
* Initial results consistent with prior literature 

---

# Next Steps

* Implement RecentPop and DecayPop
* Integrate RecBole models
* Extend experiments to additional datasets
* Analyze trendiness and popularity bias

---

# Summary

This project bridges simple popularity-based recommendation methods and modern recommender systems by combining:

* reproducible experimentation
* time-aware modeling
* cross-domain evaluation

The goal is to better understand the role of **popularity and time in recommendation systems** and to provide a more realistic baseline for future research.
