# Dataset Documentation

This document describes all datasets used in the thesis project, their role, preprocessing, and how they are integrated into the experimental pipeline.

---

# 1. Top-N Recommendation Datasets

## 1.1 MovieLens 20M

### Purpose

MovieLens 20M serves as the primary dataset for the Top-N recommendation setting. It is used for:

- initial prototypical experiments with popularity-based models
- integration and evaluation of models within the RecBole framework
- controlled comparison between time-aware popularity models and model-based approaches

---

### Source

GroupLens Research

---

### Dataset Version / Time Reference

- interaction period ends in 2015  
- dataset release: 2016  
- the version used corresponds to the downloaded dataset at the beginning of the project  

---

### Local Path

`data/raw/movielens/`

---

### Expected Main File

`ratings.csv`

---

### Relevant Fields

- `userId`
- `movieId`
- `timestamp`

---

### Preprocessing

The dataset is transformed into a unified implicit feedback format:

- mapping to: `user_id`, `item_id`, `timestamp`
- chronological sorting per user
- removal of users with fewer than 2 interactions

---

### Data Splitting

A chronological leave-one-out split is applied:

- last interaction per user → test set  
- all previous interactions → training set  

This ensures a realistic temporal evaluation scenario.

---

### RecBole Integration

The dataset is converted into RecBole format:

- `.inter` file for RecBole input  
- consistent field mapping for user, item, and time  

Example:

`data/recbole/movielens_recbole/movielens_recbole.inter`

---

### Role in Thesis

MovieLens is used as:

- the primary benchmark dataset
- the main environment for developing and validating RecBole-based model implementations
- the reference dataset for comparing:
  - MostPop
  - RecentPop
  - DecayPop
  - model-based methods (e.g., BPR)

---

### Notes

Initial experiments were conducted using a custom evaluation pipeline.  
In the final setup, all models are integrated and evaluated within the RecBole framework to ensure consistency and reproducibility.

---

## 1.2 Amazon Reviews

### Purpose

Amazon Reviews are used as a second domain to evaluate the generalization of findings beyond the movie recommendation setting.

---

### Source

Amazon Review Data (McAuley Lab)

---

### Dataset Version / Time Reference

- downloaded dataset version: 2023  
- used as a large-scale product recommendation dataset  

---

### Local Path

`data/raw/amazon/`

---

### Data Format

JSONL format (e.g., `Video_Games.jsonl`)

---

### Relevant Fields

- `user_id`
- `asin` → mapped to `item_id`
- `timestamp`

---

### Preprocessing

- conversion to implicit interaction format  
- mapping to `user_id`, `item_id`, `timestamp`  
- chronological sorting  
- filtering of users with insufficient interactions  

---

### Data Splitting

Same as MovieLens:

- chronological leave-one-out split  

---

### RecBole Integration

- conversion into `.inter` format  
- alignment with RecBole input requirements  

---

### Role in Thesis

Amazon is used for:

- cross-domain validation  
- testing robustness of models under:
  - higher sparsity  
  - stronger popularity bias  
- evaluating whether time-aware popularity models generalize beyond controlled datasets  

---

# 2. Session-Based Recommendation Datasets

## 2.1 Yoochoose

### Purpose

Used for session-based recommendation in an e-commerce context.

---

### Source

RecSys Challenge 2015

---

### Local Path

`data/raw/yoochoose/`

---

### Relevant Fields

- `session_id`
- `item_id`
- `timestamp`

---

### Role in Thesis

- evaluation of session-based recommendation methods  
- comparison with Top-N approaches  
- analysis of short-term user intent  

---

### Planned Integration

- conversion into session-based RecBole-compatible format  
- evaluation using session-based models (e.g., KNN-based approaches)

---

## 2.2 Globo

### Purpose

Session-based news recommendation dataset with strong temporal dynamics.

---

### Role

- evaluation of time-sensitive recommendation  
- comparison with Yoochoose (different domain behavior)

---

## 2.3 Adressa

### Purpose

Additional dataset for news recommendation.

---

### Role

- complementary dataset for temporal and session-based analysis  
- optional extension for robustness evaluation  

---

# 3. General Notes

## Experimental Setup

The project is structured into two main phases:

### Phase 1 – Prototypical Pipeline

- initial implementation of:
  - MostPop
  - RecentPop
  - DecayPop  
- custom evaluation pipeline  
- used for validation of model logic and baseline comparisons  

---

### Phase 2 – RecBole-Based Framework

- full integration of models into RecBole  
- implementation of custom models:
  - MostPop
  - RecentPop
  - DecayPop  
- standardized evaluation within RecBole  
- comparison with built-in models (e.g., BPR)

---

## Design Principles

- unified data representation (`user_id`, `item_id`, `timestamp`)
- chronological evaluation
- reproducibility
- framework-based extensibility

---

## Current Status

- Top-N pipeline implemented and validated  
- Amazon dataset integrated  
- RecBole integration started  
- transition towards fully framework-based implementation ongoing  