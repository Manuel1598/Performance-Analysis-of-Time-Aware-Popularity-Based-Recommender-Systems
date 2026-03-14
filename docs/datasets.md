# Dataset Documentation

This file documents all datasets used in the thesis project, including their source, local storage path, and intended purpose.

---

## 1. MovieLens 20M

### Purpose
Used as the primary dataset for reproducing the reference paper.

### Source
GroupLens Research

### Local Path
`data/raw/movielens/`

### Expected Main File
`ratings.csv`

### Relevant Fields
- `userId`
- `movieId`
- `timestamp`

### Role in Thesis
- reproduction of the baseline study
- first benchmark for MostPop, RecentPop, and DecayPop

---

## 2. Amazon Reviews

### Purpose
Used as a second domain to test whether the findings generalize beyond movie recommendation.

### Source
Amazon Review Data / McAuley Lab

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