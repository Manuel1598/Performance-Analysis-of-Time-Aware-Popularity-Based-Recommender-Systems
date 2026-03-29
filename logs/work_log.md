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


## 2026-03-18 – Repository refactoring and structure update

* Reorganized the project structure into dedicated submodules (e.g., `models/mostpop`, `models/recentpop`)
* Adjusted file paths across scripts to match the new folder structure
* Fixed project root resolution using `Path(...).parents[...]` where necessary
* Ensured all preprocessing, recommendation, and evaluation scripts run correctly after refactoring

### Why this was done

As the project grows to include multiple models and datasets, a clear modular structure is required.
This improves maintainability, readability, and scalability of the codebase.
The refactoring ensures that future models (e.g., DecayPop, RecBole-based methods) can be integrated cleanly.



## 2026-03-27 – RecentPop baseline implementation (MovieLens)

* Created `recentpop.py`
* Implemented time-aware popularity computation using a sliding time window (30 days)
* Computed item popularity relative to each user's test timestamp
* Filtered already seen items per user
* Generated top-10 recommendations for all MovieLens test users
* Saved recommendations to `results/movielens_recentpop_recommendations.csv`

### Why this was done

This step extends the static MostPop baseline by incorporating temporal dynamics.
Instead of using all historical interactions, RecentPop only considers interactions within a recent time window before the recommendation time.
This allows the model to better capture short-term popularity trends.



## 2026-03-27 – RecentPop evaluation (MovieLens)

* Created `evaluate_recentpop.py`
* Loaded MovieLens test dataset and RecentPop recommendation output
* Built ground-truth mapping from test interactions
* Constructed ranked recommendation lists per user
* Computed evaluation metrics: HR@5, HR@10, NDCG@5, NDCG@10
* Evaluated recommendations for all test users
* Saved evaluation results to `results/movielens_recentpop_metrics.csv`

### Results
- HR@5: 0.0558
- HR@10: 0.0903
- NDCG@5: 0.0365
- NDCG@10: 0.0475

### Why this was done

This step evaluates the time-aware RecentPop model using standard ranking metrics.
The results enable a direct comparison with the MostPop baseline.
This comparison is essential for analyzing the impact of temporal information on recommendation performance.


## 2026-03-29 – DecayPop baseline implementation (MovieLens)

* Created `decaypop.py`
* Implemented time-aware popularity computation using exponential time decay
* Computed weighted item popularity relative to each user's test timestamp
* Filtered already seen items per user
* Generated top-10 recommendations for all MovieLens test users
* Saved recommendations to `results/movielens_decaypop_recommendations.csv`

### Why this was done

This step extends the popularity-based models by introducing a continuous time-aware weighting mechanism.
Unlike MostPop, which treats all past interactions equally, and RecentPop, which uses a fixed time window, DecayPop gradually reduces the influence of older interactions.
This allows the model to capture temporal dynamics in a smoother and more realistic way.



## 2026-03-29 – DecayPop evaluation (MovieLens)

* Created `evaluate_decaypop.py`
* Loaded MovieLens test dataset and DecayPop recommendation output
* Built ground-truth mapping from test interactions
* Constructed ranked recommendation lists per user
* Computed evaluation metrics: HR@5, HR@10, NDCG@5, NDCG@10
* Evaluated recommendations for all test users
* Saved evaluation results to `results/movielens_decaypop_metrics.csv`

### Results

* HR@5: 0.0424
* HR@10: 0.0714
* NDCG@5: 0.0271
* NDCG@10: 0.0364

### Why this was done

This step evaluates the DecayPop model using the same evaluation pipeline as MostPop and RecentPop.
The results enable a direct comparison between static popularity, recent-window popularity, and continuous time-decayed popularity.
This comparison is central to understanding how different temporal modeling strategies influence recommendation performance.



## 2026-03-29 – Current Status (End of MovieLens experiments)

* Implemented and evaluated three popularity-based models:

  * MostPop
  * RecentPop
  * DecayPop
* Established a complete experimental pipeline:

  * preprocessing
  * chronological splitting
  * recommendation generation
  * evaluation
* Obtained comparable results across all models on MovieLens

### Why this is important

This milestone completes the initial Top-N experimental setup.
It provides the foundation for analyzing the impact of temporal information and for extending the study to additional datasets and model types.


## 2026-03-29 – Comparative analysis of popularity-based models (MovieLens)

* Created a comparison script for MovieLens popularity-based models
* Combined evaluation metrics for MostPop, RecentPop, and DecayPop
* Saved a consolidated comparison table to `results/analysis_results/`
* Generated visual comparisons for HR@10 and NDCG@10
* Verified that time-aware models outperform the static MostPop baseline

### Why this was done

This step consolidates the results of the three implemented popularity-based models into a directly comparable format.
It enables a structured analysis of how temporal information influences recommendation quality and provides the basis for the Results and Discussion sections of the thesis.
