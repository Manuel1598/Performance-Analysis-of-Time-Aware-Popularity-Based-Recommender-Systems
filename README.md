# Performance Analysis of Time-Aware Popularity-Based Recommender Systems

**Master Thesis Project**  
Author: Manuel Weilguni  
University: AAU  
Year: 2026  

---

# Project Overview

This repository contains the implementation and experiments for the master thesis:

**"Performance Analysis of Time-Aware Popularity-Based Recommender Systems"**

The goal of this work is to analyze the performance of popularity-based recommendation methods, with a particular focus on:

- temporal dynamics  
- trendiness  
- popularity bias  

The project follows a **framework-based approach using RecBole**, where both existing and custom models are implemented and evaluated in a unified environment.

---

# Research Motivation

Popularity-based recommender systems are widely used as baselines due to their simplicity and strong empirical performance.

The standard baseline, **MostPop**, ranks items based on global interaction frequency. However, this approach ignores that:

- popularity evolves over time  
- items may only be relevant within specific time windows  
- evaluation without temporal awareness can lead to misleading conclusions  

This thesis revisits popularity-based recommendation and investigates whether **time-aware extensions provide stronger and more realistic baselines**, especially when compared to model-based methods.

---

# Research Objectives

The main objectives of this thesis are:

- Implement popularity-based models:
  - MostPop  
  - RecentPop  
  - DecayPop  

- Integrate these models as **custom implementations within the RecBole framework**

- Compare them with model-based recommender systems:
  - BPR  
  - NeuMF  
  - SVD  

- Extend the analysis to **session-based recommendation**:
  - GRU4Rec  

- Analyze:
  - temporal effects  
  - popularity bias  
  - fairness aspects  

- Evaluate generalization across multiple datasets and domains  

---

# Research Questions

> How does incorporating temporal information affect the performance and behavior of popularity-based recommender systems?

Sub-questions:

- Do time-aware popularity models improve ranking performance?  
- How do they compare to model-based approaches?  
- How does popularity bias influence results?  
- Do findings generalize across datasets?  
- How do results differ between Top-N and session-based settings?  

---

# Methodology

The project follows a two-phase approach:

## Phase 1 – Prototype Pipeline

- standalone implementation of popularity-based models  
- custom evaluation pipeline  
- validation of model logic and reproducibility  

## Phase 2 – RecBole-Based Framework (Main System)

- integration of all models into RecBole  
- implementation of custom models:
  - MostPop  
  - RecentPop  
  - DecayPop  

- standardized training and evaluation  
- direct comparison with RecBole baselines  

---

# Datasets

## Top-N Recommendation

- MovieLens (primary benchmark)  
- Amazon Reviews (cross-domain validation)  

## Session-Based Recommendation

- Yoochoose (e-commerce)  
- Globo (news recommendation)  
- Adressa (optional news dataset)  

---

# Implemented Models

## Custom Models (RecBole)

- MostPop  
- RecentPop  
- DecayPop  

These models are implemented as **native RecBole models**.

---

## Baselines (RecBole)

- BPR (Bayesian Personalized Ranking)  
- NeuMF (Neural Matrix Factorization)  
- SVD  
- GRU4Rec (session-based)  

---

# Evaluation

Evaluation is performed using RecBole’s ranking-based metrics:

- NDCG@k  
- MRR@k  

Additionally analyzed:

- coverage  
- popularity bias  
- distribution of recommended items  

---

# Project Structure

The project is organized into:

- `src/` → source code  
- `data/` → datasets  
- `results/` → outputs and metrics  
- `docs/` → documentation  
- `logs/` → work log  

The codebase separates:

- prototype implementation (exploratory phase)  
- RecBole-based implementation (final system)  

---

# Reproducibility

The project is designed for full reproducibility:

- documented preprocessing  
- deterministic data splits  
- unified evaluation within RecBole  

See:

- `docs/`  
- `logs/`  
- `data/`  

---

# Current Status

Completed:

- prototype pipeline for Top-N recommendation  
- implementation of MostPop, RecentPop, DecayPop  
- integration of MovieLens and Amazon datasets  

Ongoing:

- RecBole integration of custom models  
- implementation of MostPop as first RecBole-native model  

---

# Next Steps

- implement MostPop in RecBole  
- extend to RecentPop and DecayPop  
- integrate additional RecBole baselines  
- extend to session-based recommendation  
- perform systematic hyperparameter tuning  

---

# Summary

This project transitions from a standalone experimental setup to a **framework-based evaluation using RecBole**, enabling:

- standardized comparison  
- reproducibility  
- extensibility  

The goal is to better understand the role of:

- popularity  
- time  
- model complexity  

in modern recommender systems.