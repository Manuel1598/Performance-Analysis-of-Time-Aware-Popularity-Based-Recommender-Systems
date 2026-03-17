# Work Log

## 2026-03-14 – Project setup
- Created initial project folder structure
- Initialized Git repository
- Added `.gitignore`
- Added `.gitkeep` files for tracked folder structure
- Created initial README
- Prepared repository for reproducible thesis development

### Why this was done
The goal of this step was to create a clean and reproducible project structure.
Raw data, processed data, source code, logs, and results were separated from the beginning.
This makes the project easier to understand, document, and reproduce later.


## 2026-03-17 – MovieLens preprocessing
- Created `src/preprocessing_movielens.py`
- Loaded the MovieLens ratings file
- Reduced the dataset to `user_id`, `item_id`, and `timestamp`
- Renamed columns to a unified schema
- Sorted all interactions chronologically per user
- Saved the processed interaction file to `data/processed/movielens_interactions.csv`

### Why this was done
This step creates a clean and unified interaction format for the recommender experiments.
Only the information required for the popularity-based baselines was kept.
Chronological ordering is necessary for the later leave-one-out split and time-aware evaluation.


## 2026-03-17 – MovieLens leave-one-out split
- Created `src/split.py`
- Loaded the processed MovieLens interaction dataset
- Sorted interactions by `user_id` and `timestamp`
- Created a chronological leave-one-out split
- Assigned the last interaction of each user to the test set
- Assigned all previous interactions of each user to the training set
- Saved the resulting files as `movielens_train.csv` and `movielens_test.csv`

### Why this was done
This step creates the time-aware offline evaluation setup used for the reproduction.
The split ensures that future interactions are not leaked into training and that each user has exactly one held-out test interaction.

- Added a filter to remove users with fewer than 2 interactions before the leave-one-out split

### Why this was done
Users with only one interaction would appear only in the test set and not in the training set.
Removing these users ensures that every evaluated user is present in both training and test data.

