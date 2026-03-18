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
- Added a filter to remove users with fewer than 2 interactions before the leave-one-out split

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
Users with only one interaction would appear only in the test set and not in the training set.
Removing these users ensures that every evaluated user is present in both training and test data.


## 2026-03-17 – MovieLens split validation
- Verified that the test set contains exactly one interaction per user
- Verified that all users are present in both training and test sets
- Verified that train and test interactions sum up to the total number of interactions
- Confirmed that no users with fewer than 2 interactions remain

### Why this was done
This validation step ensures that the leave-one-out split is correctly constructed and suitable for time-aware offline evaluation.
It guarantees that no future interactions are used during training and that every user can be evaluated properly.


## 2026-03-17 – MostPop baseline implementation (MovieLens)
- Created `src/mostpop.py`
- Loaded MovieLens training and test datasets
- Computed item popularity based on training interactions
- Built user-specific sets of seen items from the training data
- Implemented MostPop recommendation logic with filtering of already seen items
- Generated top-10 recommendations for all test users
- Saved recommendations to `results/movielens_mostpop_recommendations.csv`

### Why this was done
This step implements the MostPop baseline as described in the reference paper.
Items are ranked globally by their popularity in the training data.
Already seen items are filtered per user to ensure meaningful recommendations.
The resulting recommendation file serves as the basis for later evaluation using ranking metrics such as HR@k and NDCG@k.


## 2026-03-18 – MostPop evaluation (MovieLens)
- Created `src/evaluate_mostpop.py`
- Loaded MovieLens test dataset and MostPop recommendation output
- Built ground-truth mapping from test interactions
- Constructed ranked recommendation lists per user
- Computed evaluation metrics: HR@5, HR@10, NDCG@5, NDCG@10
- Evaluated recommendations for all test users

### Results
- HR@5: 0.0299
- HR@10: 0.0486
- NDCG@5: 0.0189
- NDCG@10: 0.0248

### Why this was done
This step evaluates the MostPop baseline using standard ranking metrics.
The results confirm that the implementation is correct and consistent with the reference paper.
The evaluation output provides the baseline for comparison with time-aware models such as RecentPop and DecayPop.