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
- Created `src/datapipeline/preprocessing_movielens.py`
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
- Created `src/datapipeline/split.py`
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
- Created `src/models/mostpop/mostpop.py`
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
- Created `src/models/mostpop/evaluate_mostpop.py`
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
- MRR@5: 0.0152
- MRR@10: 0.0177
- Coverage: 0.0154

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
- MRR@5: 0.0302
- MRR@10: 0.0347
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

### Results
- RecentPop outperforms all models
- DecayPop improves over MostPop but underperforms compared to RecentPop

### Why this was done
This step consolidates the results of the three implemented popularity-based models into a directly comparable format.
It enables a structured analysis of how temporal information influences recommendation quality and provides the basis for the Results and Discussion sections of the thesis.

## 2026-04-08 – Shared utility refactoring for popularity-based models

* Created `src/utils/io.py`
* Centralized shared input/output functionality:
  * `load_data(...)`
  * `save_recommendations(...)`
  * `save_results(...)`
* Created `src/utils/recommendation.py`
* Centralized shared recommendation-related helper functions:
  * `build_user_seen_items(...)`
  * `build_ground_truth(...)`
  * `build_recommendation_lists(...)`
* Updated `mostpop.py`, `recentpop.py`, and `decaypop.py` to use the shared utility modules
* Updated all evaluation scripts to use shared utility modules
* Standardized project root resolution using `Path(__file__).resolve().parents[3]`

### Why this was done

The three popularity-based models originally contained duplicated logic for data loading, recommendation output generation, and evaluation preparation.
Centralizing these functions reduces redundancy and makes the code easier to maintain and extend.
This refactoring is an important preparation step for integrating additional models and frameworks later while keeping a consistent experimental pipeline.


## 2026-04-08 – Centralized ranking evaluation pipeline

* Created `src/evaluation/metrics.py`
* Moved shared ranking metric implementations into a central module:
  * `hit_rate_at_k(...)`
  * `ndcg_at_k(...)`
  * `mrr_at_k(...)`
* Created `src/evaluation/evaluator.py`
* Implemented shared evaluation function:
  * `evaluate_recommendations(...)`
* Updated `evaluate_mostpop.py`, `evaluate_recentpop.py`, and `evaluate_decaypop.py` to use the centralized evaluation pipeline
* Confirmed that all three evaluation scripts still run correctly after refactoring

### Why this was done

The evaluation logic was previously duplicated across all popularity-based models.
Centralizing the ranking metrics and evaluation routine improves consistency and reduces the risk of implementation differences between models.
This step also strengthens the reproducibility of the thesis experiments and prepares the codebase for future evaluation extensions such as additional cutoffs, coverage metrics, or trendiness analysis.


## 2026-04-09 – Migration of popularity models to an object-oriented framework

* Introduced `src/models/base.py` with a shared recommender base interface
* Created a new modular popularity model structure under `src/models/popularity/`
* Implemented:
  * `MostPopRecommender`
  * `RecentPopRecommender`
  * `DecayPopRecommender`
* Added dedicated runner scripts:
  * `run_mostpop.py`
  * `run_recentpop.py`
  * `run_decaypop.py`
* Reused the centralized utility and evaluation modules introduced earlier
* Verified that the new object-oriented implementations reproduce the same recommendation outputs and evaluation results as the previous function-based versions

### Why this was done

The original popularity-based implementations were script-oriented and model-specific.
Migrating them to a shared object-oriented structure creates a cleaner and more extensible experimental framework.
This is an important preparation step for integrating additional recommender models, including RecBole-based methods, while keeping a unified pipeline for training, recommendation generation, and evaluation.



## 2026-04-10 – RecBole integration and first BPR baseline (MovieLens)

* Added `recbole==1.2.0` to the project environment
* Created `src/models/recbole/prepare_recbole_movielens.py`
* Converted `data/processed/movielens_train.csv` into RecBole atomic format
* Generated `data/recbole/movielens_recbole/movielens_recbole.inter`
* Created `src/models/recbole/bpr_wrapper.py`
* Implemented a RecBole-based BPR recommender wrapper compatible with the shared project interface
* Created `src/models/recbole/run_bpr.py`
* Generated `results/movielens_bpr_recommendations.csv`
* Created `src/models/recbole/evaluate_bpr.py`
* Evaluated BPR using the shared evaluation pipeline
* Generated `results/movielens_bpr_metrics.csv`

### Results

* HR@5: 0.0423
* HR@10: 0.0718
* NDCG@5: 0.0266
* NDCG@10: 0.0360
* MRR@5: 0.0214
* MRR@10: 0.0253

### Why this was done

This step introduces the first model-based baseline into the experimental framework using RecBole.
The goal is to compare the previously implemented popularity-based models with a classical collaborative filtering approach under the same Top-N evaluation setup.
The results provide the first direct comparison between simple time-aware popularity methods and a learned recommendation model on MovieLens.


## 026-04-10 – Integration and evaluation of BPR (RecBole) baseline (MovieLens)
* Integrated RecBole into the project pipeline
* Implemented a wrapper for the BPR (Bayesian Personalized Ranking) model
* Converted MovieLens data into RecBole-compatible format
* Trained BPR model and generated top-10 recommendations for all test users
* Evaluated BPR using the existing Top-N evaluation pipeline
* Extended the analysis to include BPR in the comparison with MostPop, RecentPop, and DecayPop
* Updated comparison tables and plots (HR@10, NDCG@10, MRR@10)

### Results
* BPR achieves performance comparable to DecayPop
* RecentPop remains the best-performing model
* Static MostPop baseline performs worst


### Why this was done

This step extends the evaluation by including a model-based collaborative filtering baseline.
It enables a direct comparison between simple popularity-based approaches and a learned recommendation model.
The results show that time-aware popularity models can match or outperform BPR in this setting, highlighting the importance of temporal dynamics in recommendation tasks.


## 2026-04-15 – Extension of Top-N evaluation with coverage metric (MovieLens)

* Extended the evaluation pipeline by adding the Coverage metric
* Implemented a new function to compute catalog coverage based on recommended items
* Updated the shared evaluator to include Coverage alongside HR@k, NDCG@k, and MRR@k
* Adapted all evaluation scripts (MostPop, RecentPop, DecayPop, BPR) to compute Coverage
* Extended the comparison analysis to include Coverage in the result table and plots
* Generated an additional visualization comparing Coverage across all models

### Results

* MostPop shows the lowest Coverage, indicating strong concentration on few popular items
* RecentPop and DecayPop increase Coverage compared to MostPop
* BPR achieves the highest Coverage by a large margin

### Why this was done

This step extends the evaluation beyond accuracy-based metrics.
While HR, NDCG, and MRR measure ranking quality, Coverage reflects how broadly the item catalog is utilized.

Including Coverage enables a more comprehensive comparison between popularity-based and model-based approaches.
It highlights the trade-off between recommendation accuracy and diversity of recommended items.


## 2026-04-18 – Amazon Video Games preprocessing and chronological split

* Added `src/datapipeline/preprocessing_amazon.py`
* Loaded the Amazon Video Games dataset from JSONL format
* Extracted the fields `user_id`, `parent_asin`, and `timestamp`
* Normalized timestamps from milliseconds to seconds
* Converted the raw data into the unified interaction format:
  * `user_id`
  * `item_id`
  * `timestamp`
* Sorted interactions chronologically per user
* Filtered users with fewer than 2 interactions
* Saved the processed file as `data/processed/amazon_interactions.csv`

* Added `src/datapipeline/split_amazon.py`
* Created a chronological leave-one-out split for Amazon
* Assigned the last interaction of each user to the test set
* Assigned all previous interactions of each user to the training set
* Saved the resulting files as:
  * `data/processed/amazon_train.csv`
  * `data/processed/amazon_test.csv`
* Validated the split:
  * exactly one test interaction per user
  * identical user sets in train and test
  * train and test sizes match the original interaction count
  * no chronology violations detected

### Why this was done

This step extends the Top-N recommendation pipeline to a second domain beyond MovieLens.
The Amazon Video Games dataset is used to test whether the findings from MovieLens also generalize to a different recommendation setting.
Using the same preprocessing and chronological split strategy ensures methodological consistency across datasets.


## 2026-04-18 – Generalized MostPop pipeline for MovieLens and Amazon

* Refactored the MostPop runner into a shared multi-dataset version
* Added dataset-specific configuration for MovieLens and Amazon
* Extended the shared recommendation pipeline to support both numeric and string-based user/item identifiers
* Verified that MovieLens MostPop still reproduces the same results as before
* Ran MostPop on the Amazon Video Games dataset using the same Top-N evaluation setup
* Generated recommendation outputs and evaluation results for both datasets

### Results

* MovieLens MostPop results remained unchanged after refactoring
* Amazon MostPop produced substantially lower ranking accuracy than MovieLens
* Amazon MostPop also showed extremely low Coverage, indicating a very strong concentration on a small number of popular items

### Why this was done

This step extends the Top-N recommendation pipeline from a single dataset to a multi-dataset setup.
By generalizing the runner and evaluation structure, the same baseline model can now be applied consistently across domains.
This is important for testing whether the findings from MovieLens also generalize to Amazon.





## 2026-04-21 – Generalized MostPop pipeline for MovieLens and Amazon

* Refactored the MostPop runner into a shared multi-dataset version  
* Added dataset-specific configuration for MovieLens and Amazon  
* Extended the shared recommendation pipeline to support both numeric and string-based user/item identifiers  
* Verified that MovieLens MostPop still reproduces the same results as before  
* Ran MostPop on the Amazon Video Games dataset using the same Top-N evaluation setup  
* Generated recommendation outputs and evaluation results for both datasets  

### Results

* MovieLens MostPop results remained unchanged after refactoring  
* Amazon MostPop produced substantially lower ranking accuracy than MovieLens  
* Amazon MostPop also showed extremely low Coverage, indicating a very strong concentration on a small number of popular items  

### Interpretation

The results confirm that:

* popularity-based recommendations behave differently across domains  
* Amazon exhibits a stronger popularity bias and higher sparsity  
* simple global popularity is less effective in large-scale, sparse datasets  

---

### Strategic Update

Following a project discussion, the overall direction of the thesis was refined:

* Instead of maintaining a standalone recommendation pipeline, the project will transition to a **framework-based approach using RecBole**  
* All models, including MostPop, RecentPop, and DecayPop, will be **re-implemented as native RecBole models**  
* RecBole will serve as the central framework for training, evaluation, and comparison  

---

### Role of This Step

This implementation represents the **final stage of the prototype pipeline**:

* validated correctness of the MostPop implementation  
* ensured consistency across multiple datasets  
* confirmed reproducibility of results  
* provided a reference baseline for later RecBole integration  

---

### Next Steps

* start implementation of custom RecBole models  
* implement MostPop as first RecBole-native model  
* gradually migrate RecentPop and DecayPop into RecBole  
* shift evaluation from custom pipeline to RecBole framework  





## 2026-04-21 – First successful RecBole integration of MostPop

* Implemented MostPop as a native RecBole model (`MostPopRecBole`)
* Followed the RecBole model interface by implementing:
  * `__init__`
  * `calculate_loss`
  * `predict`
  * `full_sort_predict`
* Adapted the model to work without trainable parameters by introducing a minimal dummy parameter for compatibility with the RecBole optimizer
* Built a first RecBole runner using:
  * `Config`
  * `create_dataset`
  * `data_preparation`
  * `Trainer`
* Successfully loaded the MovieLens dataset in RecBole format (`.inter`)
* Executed a full RecBole pipeline run including training and evaluation
* Resolved compatibility issues related to:
  * dataset path handling
  * PyTorch checkpoint loading
  * non-trainable model structure

### Results

* RecBole successfully executed the full pipeline with the custom MostPop model  
* Evaluation metrics were produced using RecBole’s internal evaluation framework  
* The model behaves as expected within the RecBole environment  

---

### Observations

* The obtained evaluation results differ from the previous standalone implementation  
* This is due to differences in:
  * data splitting strategy (RecBole default vs. chronological leave-one-out)  
  * evaluation setup  
* The current RecBole configuration uses a random split, which is not aligned with the thesis methodology  

---

### Interpretation

This step confirms that:

* custom popularity-based models can be fully integrated into RecBole  
* RecBole can be used as the central framework for model execution and evaluation  
* additional work is required to align the evaluation setup with the intended experimental design  

---

### Role of This Step

This step represents the **first successful transition from the prototype pipeline to the RecBole-based framework**:

* validates the technical feasibility of custom model integration  
* establishes the foundation for all further RecBole-based experiments  
* enables direct comparison with built-in RecBole models  

---

### Methodological Note

The final experimental setup does not aim to exactly replicate the earlier standalone leave-one-out pipeline.  
Instead, the project now adopts RecBole’s standardized data preparation and evaluation workflow as the main experimental framework.  
The earlier standalone implementation remains relevant as a prototype and validation step, while the final comparison is conducted under the unified RecBole setup.

---

### Next Steps

* adapt RecBole evaluation to match chronological leave-one-out splitting  
* ensure comparability with previous experimental results  
* extend RecBole implementation to:
  * RecentPop  
  * DecayPop  
* integrate additional model-based baselines (BPR, NeuMF)  






## 2026-04-22 – Implementation of RecentPop as a RecBole-native model

* Implemented RecentPop as a custom model within the RecBole framework (`RecentPopRecBole`)
* Extended the MostPop implementation by introducing a time-based filtering mechanism
* Added a configurable time window parameter (`window_days`) to restrict interactions to recent data
* Used the maximum timestamp in the dataset as a global reference point for defining the recent interaction window
* Filtered interactions to include only those within the specified time window before computing item popularity
* Maintained compatibility with RecBole's training pipeline by including a dummy trainable parameter
* Implemented all required RecBole model interface methods:
  * `__init__`
  * `calculate_loss`
  * `predict`
  * `full_sort_predict`
* Created a dedicated RecBole runner (`run_recentpop_recbole.py`) following the standardized pipeline:
  * `Config`
  * `create_dataset`
  * `data_preparation`
  * `Trainer.fit`
  * `Trainer.evaluate`
* Successfully executed the full RecBole pipeline on the MovieLens dataset

---

### Results

* The RecentPop model executed successfully within the RecBole framework
* Evaluation metrics were produced using RecBole’s internal evaluation pipeline
* Compared to MostPop, the RecentPop model achieved lower ranking performance in the current setup:
  * lower Hit@k
  * lower NDCG@k
  * lower MRR@k

---

### Observations

* The current implementation uses a **global time window**, defined relative to the maximum timestamp in the dataset
* This approach differs from more fine-grained temporal models that adapt the time window per user or per interaction
* Due to the global filtering, only a subset of interactions contributes to the popularity estimation
* This may reduce robustness, especially in sparse datasets such as MovieLens

---

### Interpretation

The results suggest that:

* a simple global RecentPop formulation may not outperform static popularity (MostPop) in all settings
* the effectiveness of time-aware popularity models strongly depends on how temporal context is defined
* the interaction between time-awareness and the evaluation setup (e.g., random split vs. chronological behavior) plays a significant role

---

### Role of This Step

This step represents the **first time-aware extension of popularity-based models within RecBole**:

* validates that temporal extensions can be integrated into the framework
* establishes a foundation for more advanced time-aware models
* enables systematic comparison between:
  * static popularity (MostPop)
  * time-window-based popularity (RecentPop)

---

### Next Steps

* implement DecayPop as a RecBole-native model
* compare MostPop, RecentPop, and DecayPop under identical RecBole settings
* refine temporal modeling strategies (e.g., dynamic windows or user-specific time references)
* analyze the impact of time-awareness on recommendation performance and popularity bias



## 2026-04-22 – Implementation of DecayPop as a RecBole-native model

* Implemented DecayPop as a custom model within the RecBole framework (`DecayPopRecBole`)
* Extended the popularity-based modeling approach by introducing a continuous time-decay function
* Replaced the hard cutoff of RecentPop with an exponential decay weighting scheme
* Defined the decay function as:

  weight = exp(-λ * Δt)

  where:
  * Δt is the time difference between an interaction and the most recent timestamp
  * λ is a configurable decay parameter (`decay_lambda`)

* Used the maximum timestamp in the dataset as a global reference point for computing time differences
* Weighted all interactions based on recency instead of discarding older interactions
* Maintained compatibility with RecBole’s training pipeline by including a dummy trainable parameter
* Implemented all required RecBole model interface methods:
  * `__init__`
  * `calculate_loss`
  * `predict`
  * `full_sort_predict`
* Created a dedicated RecBole runner (`run_decaypop_recbole.py`)
* Successfully executed the full RecBole pipeline on the MovieLens dataset

---

### Results

* The DecayPop model executed successfully within the RecBole framework
* Evaluation metrics were produced using RecBole’s internal evaluation pipeline
* Results will be compared against:
  * MostPop (static popularity)
  * RecentPop (time-window-based popularity)

---

### Observations

* Unlike RecentPop, DecayPop considers all interactions but assigns lower weights to older ones
* This results in a smoother and more stable popularity estimation
* The model avoids abrupt changes caused by hard time windows
* The behavior of the model is highly sensitive to the decay parameter λ

---

### Interpretation

The DecayPop formulation provides a more flexible representation of temporal dynamics:

* it captures gradual changes in item popularity
* it balances long-term popularity and short-term trends
* it may perform more robustly than RecentPop in sparse or long-tailed datasets

---

### Role of This Step

This step completes the implementation of the three core popularity-based models within RecBole:

* MostPop (static)
* RecentPop (window-based)
* DecayPop (time-decay-based)

This enables a fully consistent and framework-based comparison of different popularity formulations.

---

### Next Steps

* compare MostPop, RecentPop, and DecayPop under identical RecBole settings
* analyze the effect of time-awareness on recommendation performance
* investigate the impact of decay parameter choices
* extend experiments to additional datasets (e.g., Amazon)


## 2026-04-22 – Comparative evaluation of RecBole models on MovieLens

* Extended the RecBole comparison pipeline to include the built-in BPR model
* Executed all four models under the same RecBole-based experimental setup:
  * MostPop
  * RecentPop
  * DecayPop
  * BPR
* Stored individual metric outputs and generated a consolidated comparison table
* Created comparison plots for:
  * Hit@10
  * NDCG@10
  * MRR@10

### Results

* BPR achieved the strongest performance across all measured ranking metrics
* Among the popularity-based models, MostPop performed best
* DecayPop consistently outperformed RecentPop
* Neither RecentPop nor DecayPop surpassed MostPop in the current RecBole setup

### Interpretation

The current results suggest that:

* the model-based BPR approach benefits from personalized latent preference learning
* static global popularity remains a strong simple baseline
* adding temporal popularity information in a simple global form does not automatically improve recommendation quality
* smooth temporal weighting (DecayPop) is more effective than a strict recent-window formulation (RecentPop)

### Role of This Step

This step establishes the first complete RecBole-based comparison between custom popularity-based models and a built-in model-based baseline on MovieLens.
It provides the foundation for extending the same comparison setup to additional datasets such as Amazon.



## 2026-04-24 – Amazon integration for RecBole MostPop experiments

* Prepared the Amazon Video Games dataset for RecBole-based experiments
* Converted the processed Amazon interaction data into RecBole `.inter` format
* Added support for running the RecBole MostPop model on multiple datasets
* Extended the MostPop RecBole runner from MovieLens-only execution to a multi-dataset setup
* Successfully executed MostPop on the Amazon dataset using the RecBole pipeline
* Stored the Amazon MostPop evaluation output in `recbole_results/amazon_mostpop_recbole_metrics.csv`

---

### Results

Amazon MostPop results:

* Hit@5: 0.0161
* Hit@10: 0.0255
* NDCG@5: 0.0104
* NDCG@10: 0.0134
* MRR@5: 0.0086
* MRR@10: 0.0099

---

### Observations

* The Amazon results are substantially lower than the corresponding MovieLens results
* This is expected due to the higher sparsity and larger number of users and items in the Amazon dataset
* The result confirms that the RecBole setup can now be applied beyond MovieLens

---

### Role of This Step

This step extends the RecBole-based experimental setup from a single dataset to a cross-domain setting.

It provides the foundation for evaluating whether findings from MovieLens generalize to Amazon.

---

### Next Steps

* run RecentPop on Amazon
* run DecayPop on Amazon
* evaluate BPR on Amazon where computationally feasible
* compare MovieLens and Amazon results under the same RecBole framework





## 2026-04-27 – Amazon integration for RecentPop and DecayPop (RecBole)

* Extended the RecBole-based experimental setup to include time-aware popularity models on the Amazon dataset
* Adapted the RecentPop and DecayPop runners to support multiple datasets (MovieLens and Amazon)
* Successfully executed RecentPop on the Amazon Video Games dataset
* Successfully executed DecayPop on the Amazon Video Games dataset
* Stored evaluation outputs in:
  * `recbole_results/amazon_recentpop_recbole_metrics.csv`
  * `recbole_results/amazon_decaypop_recbole_metrics.csv`

---

### Results

Amazon RecentPop results:

* Hit@5: 0.0034
* Hit@10: 0.0075
* NDCG@5: 0.0023
* NDCG@10: 0.0036
* MRR@5: 0.0019
* MRR@10: 0.0025

Amazon DecayPop results:

* Hit@5: 0.0065
* Hit@10: 0.0112
* NDCG@5: 0.0035
* NDCG@10: 0.0050
* MRR@5: 0.0026
* MRR@10: 0.0032

---

### Observations

* Both time-aware models perform significantly worse than MostPop on the Amazon dataset
* RecentPop shows the lowest performance across all evaluated models
* DecayPop performs better than RecentPop, but still remains below MostPop
* The fixed time window in RecentPop leads to a strong loss of interaction data in sparse datasets
* Decay-based weighting mitigates this issue but does not fully recover performance

---

### Interpretation

The results highlight that:

* aggressive time filtering (RecentPop) is not suitable for highly sparse datasets like Amazon
* gradual time decay (DecayPop) is more robust but still sensitive to sparsity
* global popularity remains a strong baseline in large-scale, sparse recommendation scenarios

---

### Role of This Step

This step completes the evaluation of all popularity-based models on the Amazon dataset within the RecBole framework.

It enables a direct comparison between static and time-aware popularity approaches across domains.

---

### Next Steps

* integrate and evaluate BPR on the Amazon dataset
* extend comparison analysis to include all models across both datasets
* analyze cross-domain differences between MovieLens and Amazon



## 2026-04-25 – Integration and evaluation of BPR across datasets (RecBole)

* Implemented a unified RecBole runner for the BPR model supporting multiple datasets
* Integrated BPR into the same experimental pipeline as the popularity-based models
* Executed BPR on MovieLens and Amazon datasets using consistent evaluation settings
* Stored evaluation outputs in:
  * `recbole_results/movielens_bpr_recbole_metrics.csv`
  * `recbole_results/amazon_bpr_recbole_metrics.csv`
* Extended the analysis scripts to include BPR in model comparisons

---

### Results (MovieLens reference)

BPR results outperform all popularity-based models on MovieLens:

* highest Hit@k, NDCG@k, and MRR@k across all evaluated models
* confirms the advantage of personalized, model-based approaches in dense datasets

---

### Observations

* BPR benefits from learning user-specific preferences, unlike popularity-based models
* Performance gains are particularly visible in ranking-based metrics (NDCG, MRR)
* The RecBole integration allows direct and fair comparison under identical evaluation conditions

---

### Interpretation

The results confirm that:

* model-based approaches can outperform popularity-based baselines in structured datasets
* however, the gap depends strongly on dataset characteristics such as sparsity and interaction density
* popularity-based models remain competitive baselines, especially in large-scale scenarios

---

### Role of This Step

This step completes the integration of both popularity-based and model-based approaches within a unified RecBole framework.

It establishes the foundation for systematic model comparison across datasets.

---

### Next Steps

* analyze performance differences between popularity-based and model-based methods
* investigate the impact of sparsity and temporal dynamics on model performance
* prepare result tables and figures for the thesis




## 2026-04-26 – Cross-domain comparison of popularity-based and model-based methods

* Completed the full evaluation of all models across both MovieLens and Amazon datasets
* Integrated and executed BPR alongside MostPop, RecentPop, and DecayPop on Amazon
* Consolidated all evaluation results into a unified comparison table
* Extended analysis scripts to support cross-dataset (cross-domain) comparison
* Generated comparative metrics across models and datasets under identical evaluation settings

---

### Results

#### Amazon

* BPR:
  * Hit@10: 0.0323
  * NDCG@10: 0.0188
  * MRR@10: 0.0148

* MostPop:
  * Hit@10: 0.0255
  * NDCG@10: 0.0134
  * MRR@10: 0.0099

* DecayPop:
  * Hit@10: 0.0112
  * NDCG@10: 0.0050
  * MRR@10: 0.0032

* RecentPop:
  * Hit@10: 0.0075
  * NDCG@10: 0.0036
  * MRR@10: 0.0025

---

#### MovieLens

* BPR:
  * Hit@10: 0.3332
  * NDCG@10: 0.0758
  * MRR@10: 0.1391

* MostPop:
  * Hit@10: 0.2480
  * NDCG@10: 0.0562
  * MRR@10: 0.1053

* DecayPop:
  * Hit@10: 0.1561
  * NDCG@10: 0.0325
  * MRR@10: 0.0653

* RecentPop:
  * Hit@10: 0.1315
  * NDCG@10: 0.0252
  * MRR@10: 0.0492

---

### Observations

* All models perform significantly worse on Amazon compared to MovieLens
* BPR consistently outperforms all popularity-based models on both datasets
* The performance gap between BPR and popularity-based models is larger on MovieLens than on Amazon
* MostPop remains a strong baseline, especially on the sparse Amazon dataset
* RecentPop shows the weakest performance across both datasets, with a particularly strong degradation on Amazon
* DecayPop improves over RecentPop but does not outperform MostPop on either dataset

---

### Interpretation

The results highlight several important effects:

* **Dataset characteristics strongly influence model performance**
  * MovieLens (denser) allows models to learn stronger patterns
  * Amazon (sparser) leads to overall lower accuracy

* **Time-aware popularity models are not universally beneficial**
  * RecentPop suffers from aggressive data filtering in sparse environments
  * DecayPop mitigates this but still cannot match static popularity

* **Model-based approaches (BPR) generalize better across domains**
  * personalization provides consistent improvements
  * however, gains are reduced in highly sparse datasets

---

### Role of This Step

This step represents the **first complete cross-domain evaluation** of all implemented models.

It establishes a direct comparison between:

* popularity-based vs model-based methods
* static vs time-aware approaches
* dense vs sparse datasets

---

### Next Steps

* extend experiments to session-based recommendation models (e.g., SessionKNN, GRU4Rec)
* perform systematic hyperparameter tuning for all models
* analyze the impact of temporal parameters (window size, decay rate)
* investigate popularity bias and coverage across datasets
* prepare final evaluation tables and figures for the thesis


## 2026-04-28 – Planning session-based RecBole integration

* Completed the first cross-domain Top-N RecBole evaluation
* Decided to extend the framework to session-based recommendation
* Identified VS-KNN and VSTAN as relevant session-based nearest-neighbor baselines
* Selected the `session-rec` repository as reference implementation
* Planned integration of selected session-based algorithms into the RecBole framework

### Motivation

Session-based nearest-neighbor algorithms are strong and interpretable baselines for session-based recommendation.
Prior work has shown that such methods can outperform more complex neural approaches in several settings.

### Planned Models

* VS-KNN
* VSTAN

### Next Steps

* inspect the reference implementations in `session-rec`
* define how session data must be represented in RecBole
* prepare a session-based dataset such as Yoochoose
* implement VS-KNN as the first RecBole-compatible session-based baseline


