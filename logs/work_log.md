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



## 2026-04-29 – Yoochoose Dataset Preparation and RecBole Integration

### Overview
- Integrated the Yoochoose dataset as the first session-based dataset in the RecBole framework  
- Converted raw clickstream data into RecBole `.inter` format with session-based structure  
- Implemented correct timestamp conversion to Unix time for temporal ordering  
- Filtered sessions with fewer than two interactions to enable next-item prediction  
- Created a session-based sampling pipeline to generate manageable subsets of the dataset  
- Generated a ~500k interaction sample for efficient experimentation  
- Validated the RecBole pipeline by running the MostPop baseline on the Yoochoose sample  

### Results

#### MostPop on Yoochoose (sample)
- Hit@5: 0.0002  
- Hit@10: 0.0003  
- NDCG@5: 0.0001  
- NDCG@10: 0.0001  
- MRR@5: 0.0001  
- MRR@10: 0.0001  

### Observations
- Performance is significantly lower than on MovieLens and Amazon  
- This is expected due to:
  - extreme sparsity  
  - short session lengths  
  - absence of long-term user preferences  
- Global popularity is not suitable for session-based recommendation tasks  

### Interpretation
The results confirm that:
- session-based recommendation requires context-aware models  
- global popularity fails in short-session environments  
- Yoochoose represents a fundamentally different recommendation setting  

### Role of This Step
This step establishes the data foundation for session-based recommendation experiments.

It ensures that:
- session data is correctly represented in RecBole  
- temporal ordering is preserved  
- scalable experimentation is possible via sampling  

### Next Steps
- implement session-based nearest-neighbor models  
- start with VS-KNN as first baseline  
- compare against popularity-based methods on Yoochoose  


---

## 2026-04-29 – Initial Implementation of VS-KNN in RecBole

### Overview
- Implemented the VS-KNN session-based nearest-neighbor algorithm as a custom RecBole model  
- Adapted session-based recommendation logic to the RecBole GeneralRecommender interface  
- Represented sessions as pseudo-users to reuse RecBole’s interaction format  
- Built session-item and item-session mappings from the dataset  
- Implemented cosine similarity between sessions based on item overlap  
- Integrated candidate session sampling to control computational complexity  
- Implemented full-sort prediction for compatibility with RecBole evaluation  

### Model Characteristics

#### VS-KNN
- uses session similarity instead of global popularity  
- recommends items from similar sessions  
- operates without training or learnable parameters  
- captures short-term user intent  

### Observations
- significantly more complex than popularity-based models  
- requires careful handling of:
  - session structure  
  - candidate selection  
  - computational efficiency  
- integration into RecBole requires adapting non-parametric models to a training-based framework  

### Role of This Step
This step introduces the first session-based recommendation model into the RecBole framework.

It marks the transition from:
- user-based recommendation → session-based recommendation  

### Next Steps
- run VS-KNN on Yoochoose sample  
- analyze performance vs MostPop baseline  
- optimize runtime and candidate sampling  
- extend implementation toward VSTAN  
- integrate session-based models into comparison pipeline  



# 2026-04-29 – Evaluation and Refinement of VS-KNN Implementation

## Overview
- Successfully executed the VS-KNN model on the Yoochoose sample dataset within the RecBole framework  
- Verified correct functionality of session similarity computation and scoring mechanism  
- Achieved strong Top-N performance compared to popularity-based baselines  
- Identified limitations of the current implementation due to the use of the GeneralRecommender interface  
- Analyzed the mismatch between RecBole’s default evaluation pipeline and session-based next-item prediction  

## Results

### VS-KNN on Yoochoose (sample) (Generalrecommender (false recommender))
- Hit@5: 0.4101  
- Hit@10: 0.4560  
- NDCG@5: 0.3609  
- NDCG@10: 0.3758  
- MRR@5: 0.3457  
- MRR@10: 0.3519  

## Observations
- VS-KNN significantly outperforms MostPop on session-based data  
- The model effectively captures short-term user intent through session similarity  
- High performance indicates that neighborhood-based methods are well-suited for session-based recommendation  
- However, the current implementation operates on full session information rather than true session prefixes  

## Methodological Limitation

The current VS-KNN implementation:
- uses complete sessions as input instead of session prefixes  
- relies on the GeneralRecommender interface, which is designed for user-based recommendation  
- does not fully align with the standard formulation of next-item prediction  

As a result, the evaluation represents an approximation of session-based recommendation rather than a fully correct sequential setup.  

## Interpretation

The results confirm that:
- session-based nearest-neighbor models are highly effective on clickstream data  
- RecBole can be extended to support non-parametric session-based methods  
- careful alignment between model design and evaluation protocol is critical  

## Role of This Step
This step validates the feasibility of integrating session-based nearest-neighbor models into RecBole.

It also highlights the need for a more principled integration aligned with sequential recommendation.  

## Next Steps
- refactor VS-KNN to use RecBole’s SequentialRecommender interface  
- adapt the model to operate on session prefixes (`item_seq`)  
- ensure proper next-item prediction setup  
- implement VSTAN on top of the improved sequential formulation  
- compare session-based models against popularity-based and BPR baselines  



## 2026-04-29 – Refactoring VS-KNN to RecBole SequentialRecommender

### Overview

* Refactored the initial VS-KNN implementation from `GeneralRecommender` to RecBole’s `SequentialRecommender` interface
* Replaced the pseudo-user/session approximation with a sequence-based formulation
* Adapted VS-KNN to operate on item sequences (`item_seq`) and predict the next item
* Built reference sessions from RecBole’s sequential interaction representation
* Preserved the non-parametric nearest-neighbor logic while aligning the model more closely with session-based recommendation
* Evaluated the sequential VS-KNN implementation on the Yoochoose sample dataset

---

### Results

VS-KNN Sequential on Yoochoose sample: (Correct Recommender)

* Hit@5: 0.3986
* Hit@10: 0.4947
* NDCG@5: 0.2870
* NDCG@10: 0.3182
* MRR@5: 0.2500
* MRR@10: 0.2629

---

### Observations

* The sequential implementation produces strong results on the Yoochoose sample
* Hit@10 improves compared to the first GeneralRecommender-based approximation
* NDCG and MRR are lower than in the earlier approximation, indicating that relevant items are often found in the top-10 but not always ranked at the very top
* The new implementation is methodologically better aligned with next-item prediction

---

### Interpretation

The results indicate that VS-KNN is a strong session-based baseline for clickstream data.

The migration to `SequentialRecommender` improves the conceptual fit with RecBole’s session/sequential recommendation setup and reduces the methodological limitations of the previous approximation.

---

### Role of This Step

This step improves the scientific validity of the VS-KNN integration by aligning it with RecBole’s sequential recommendation interface.

It establishes a stronger foundation for implementing VSTAN and other session-based baselines.

---

### Next Steps

* implement VSTAN using the sequential VS-KNN structure as a base
* compare MostPop and VS-KNN on Yoochoose
* extend the session-based comparison pipeline
* later perform hyperparameter tuning for `k`, sample size, and sequence length



## 2026-05-14 – Implementation of VSTAN within the RecBole framework

### Overview

* Implemented the VSTAN session-based nearest-neighbor model as a custom RecBole sequential recommender
* Extended the previously implemented VS-KNN model with additional temporal and weighting mechanisms
* Adapted the implementation to RecBole’s `SequentialRecommender` interface
* Built sequence-based recommendation using `item_seq` representations
* Added position-based weighting to emphasize more recent interactions within a session
* Added optional IDF weighting to reduce the influence of highly popular items
* Preserved compatibility with RecBole’s full-sort Top-N evaluation pipeline

---

### Model Characteristics

#### VSTAN

* extends VS-KNN with temporal and positional weighting
* emphasizes recent items within a session
* reduces dominance of globally popular items through IDF weighting
* operates as a non-parametric session-based nearest-neighbor recommender

---

### Observations

* VSTAN is structurally more complex than VS-KNN
* the model combines:
  * session similarity
  * sequence recency
  * item weighting
* integrating VSTAN into RecBole required adapting a non-neural nearest-neighbor method to a sequential recommendation framework

---

### Interpretation

The implementation demonstrates that RecBole can be extended beyond its standard neural recommendation models to support advanced session-based nearest-neighbor approaches.

The migration toward `SequentialRecommender` improves the methodological correctness of session-based next-item prediction.

---

### Role of This Step

This step completes the first integration of advanced session-based nearest-neighbor methods into the RecBole framework.

Together with VS-KNN, it establishes a session-based recommendation benchmark suite for Yoochoose experiments.

---

### Next Steps

* run VSTAN on Yoochoose sample
* compare MostPop, VS-KNN, and VSTAN
* analyze the effect of temporal weighting
* start systematic hyperparameter tuning
* investigate runtime-performance tradeoffs


# 2026-05-14 – Comparative Evaluation of Session-Based RecBole Models

## Overview

Conducted the first comparative evaluation of session-based recommendation models within the RecBole framework.

### Compared Models
- MostPop
- VS-KNN
- VSTAN

All models were executed on the Yoochoose sample dataset using identical RecBole evaluation settings.

Generated unified comparison tables and visualizations for:
- Hit@10
- NDCG@10
- MRR@10

Additionally, a dedicated session-model comparison analysis pipeline was added.

---

## Results

| Model   | Hit@10 | NDCG@10 | MRR@10 |
|---------|--------:|---------:|--------:|
| MostPop | 0.0003  | 0.0001   | 0.0001  |
| VS-KNN  | 0.4947  | 0.3182   | 0.2629  |
| VSTAN   | 0.5140  | 0.3280   | 0.2698  |

---

## Observations

- Session-based nearest-neighbor models dramatically outperform the global MostPop baseline.
- MostPop performs poorly because it ignores the current session context.
- VS-KNN effectively captures short-term user intent through session similarity.
- VSTAN further improves VS-KNN by incorporating:
  - positional weighting
  - temporal emphasis
  - IDF-based item weighting
- The improvements of VSTAN over VS-KNN are consistent across all ranking metrics.

---

## Interpretation

The results demonstrate that:

- session context is essential for next-item prediction in clickstream datasets
- simple popularity-based recommendation is insufficient for session-based recommendation tasks
- nearest-neighbor session models remain highly competitive baselines
- RecBole can successfully support both:
  - traditional recommendation models
  - session-based recommendation approaches

---

## Role of This Step

This step establishes the first complete session-based evaluation pipeline within the project.

It provides:
- a reproducible benchmarking setup
- direct comparability between session-based algorithms
- the foundation for future hyperparameter tuning and larger-scale experiments

---

## Next Steps

- perform hyperparameter tuning for VS-KNN and VSTAN
- evaluate runtime-performance tradeoffs
- scale experiments from Yoochoose sample to larger subsets
- optionally integrate neural sequential models such as GRU4Rec
- analyze popularity bias and coverage in session-based recommendation


# 2026-05-14 – GPU-Enabled Python Environment and CUDA Setup for RecBole Experiments

## Overview

- Installed Python 3.12 to improve compatibility with machine learning libraries
- Replaced the previous Python 3.14 environment due to incompatibilities with RecBole and CUDA-enabled PyTorch
- Created a new isolated virtual environment (`.venv`) for the project
- Installed CUDA-enabled PyTorch within the virtual environment
- Successfully enabled GPU acceleration for RecBole-based neural recommendation experiments
- Configured the development environment to use the new Python interpreter inside PyCharm

---

## Technical Setup

### Environment

- Python 3.12
- Virtual environment via `venv`
- CUDA-enabled PyTorch
- NVIDIA RTX 2070 SUPER GPU

### Installed Core Libraries

- PyTorch
- RecBole
- pandas
- numpy
- matplotlib

---

## Observations

- Python 3.14 caused compatibility issues with:
  - CUDA-enabled PyTorch wheels
  - RecBole dependencies
- Downgrading to Python 3.12 resolved the installation and compatibility problems
- GPU acceleration significantly improves runtime for neural recommendation models such as GRU4Rec

---

## Role of This Step

This step establishes a reproducible and scalable experimental environment for future recommendation experiments.

It provides the technical foundation for:

- GPU-based neural recommendation training
- large-scale session-based experiments
- automated hyperparameter tuning

---

## Next Steps

- migrate future experiments fully to the new virtual environment
- create a reproducible `requirements.txt`
- benchmark runtime differences between CPU and GPU execution
- use GPU acceleration for larger-scale tuning experiments

---

# 2026-05-14 – Initial Implementation of Automated Hyperparameter Tuning Pipeline

## Overview

- Started development of an automated hyperparameter tuning pipeline for RecBole experiments
- Designed a framework for systematic parameter exploration across multiple recommendation models
- Added support for configurable parameter grids
- Planned automatic result collection and comparison across tuning runs
- Prepared the tuning pipeline for future GPU-accelerated large-scale experiments

---

## Planned Functionality

The tuning pipeline is intended to:

- automatically execute multiple experiment configurations
- vary model-specific hyperparameters
- store evaluation results after each run
- identify best-performing parameter combinations
- support reproducible benchmarking across datasets and models

---

## Target Models

### Initial Focus

- VS-KNN
- VSTAN
- GRU4Rec

### Planned Future Extensions

- BPR
- NeuMF
- additional sequential recommendation models

---

## Planned Hyperparameters

### VS-KNN

- number of neighbors (`k`)
- candidate session sample size

### VSTAN

- number of neighbors (`k`)
- position decay
- IDF weighting
- candidate session sampling

### GRU4Rec

- hidden size
- learning rate
- batch size
- number of epochs
- dropout probability

---

## Observations

- Hyperparameter tuning becomes computationally feasible after enabling GPU acceleration
- Session-based models are highly sensitive to parameter selection
- Automated tuning is necessary for fair and reproducible comparison between neural and non-neural models

---

## Role of This Step

This step introduces the foundation for systematic model optimization and fair experimental evaluation.

It supports the overall goal of:

- reproducibility
- transparent evaluation
- scientifically rigorous comparison between recommendation approaches

---

## Next Steps

- finalize the tuning pipeline implementation
- add automatic CSV result aggregation
- support interruption-safe experiment continuation
- integrate tuning results into the evaluation workflow
- perform the first systematic tuning runs on Yoochoose sample datasets


# 2026-04-29 – Integration of GRU4Rec and Initial Hyperparameter Tuning Pipeline

## Overview

- Integrated the GRU4Rec neural sequential recommendation model from RecBole
- Added GPU acceleration support via CUDA for neural model training
- Evaluated GRU4Rec on the Yoochoose sample dataset
- Extended the session-based comparison pipeline with a neural baseline
- Started development of an automated hyperparameter tuning framework for session-based recommendation models

---

## GRU4Rec Results

### GRU4Rec on Yoochoose Sample

| Metric   | Value  |
|----------|--------:|
| Hit@5    | 0.3548  |
| Hit@10   | 0.4566  |
| NDCG@5   | 0.2486  |
| NDCG@10  | 0.2816  |
| MRR@5    | 0.2135  |
| MRR@10   | 0.2272  |

---

## Observations

- GRU4Rec substantially outperforms the MostPop baseline
- VS-KNN and VSTAN still achieve stronger ranking performance on the current Yoochoose sample setup
- GPU acceleration significantly reduced training time for neural recommendation experiments
- The results confirm that strong nearest-neighbor baselines remain highly competitive against neural approaches

---

## Hyperparameter Tuning Pipeline

Implemented an initial automated tuning framework supporting:

- repeated experiment execution
- configurable parameter grids
- automatic result aggregation
- CSV-based result storage

### Current Tuning Support

- VS-KNN
- VSTAN

### Currently Explored Parameters

- neighborhood size (`k`)
- candidate session sample size
- position decay
- IDF weighting

---

## Role of This Step

This step extends the project from:

- static model evaluation

toward:

- systematic model optimization
- reproducible hyperparameter exploration
- scalable GPU-accelerated experimentation

---

## Next Steps

- extend tuning support to GRU4Rec
- analyze best-performing parameter combinations
- scale experiments beyond the Yoochoose sample dataset
- compare runtime and recommendation quality across models



## 2026-05-14 – Initial execution of session-based hyperparameter tuning

### Overview

* Started the first automated hyperparameter tuning run for session-based RecBole models
* Extended the tuning setup to include:
  * VS-KNN
  * VSTAN
  * GRU4Rec
* Added GPU support for neural model tuning where available
* Configured the tuning script to store results incrementally after each completed run
* Started with a reduced parameter grid to validate stability before larger-scale tuning

---

### Purpose

The goal of this step is to verify that the tuning pipeline works reliably before running larger experiments.

The reduced grid helps test:

* automated experiment execution
* result collection
* CSV output generation
* model-specific parameter handling
* GPU support for GRU4Rec

---

### Planned Next Steps

* inspect the first tuning results
* identify best-performing configurations by MRR@10
* expand the parameter grid for longer server or overnight runs
* use tuning results to improve final model comparisons



# 2026-05-14 – Analysis and Evaluation of Session-Based Hyperparameter Tuning Results

## Overview

- Completed the first successful hyperparameter tuning runs for:
  - VS-KNN
  - VSTAN
  - GRU4Rec
- Added automated analysis and reporting for tuning results
- Implemented result ranking based on `MRR@10`
- Added extraction of:
  - best overall configurations
  - best configuration per model
- Exported summarized tuning results into structured CSV files for later thesis evaluation

---

## Main Findings

### Session-Based Nearest-Neighbor Models

- VSTAN achieved the strongest ranking performance on the Yoochoose sample dataset
- VS-KNN remained highly competitive and consistently outperformed the popularity baseline

### GRU4Rec

- Initial GRU4Rec configurations produced weak results due to incorrect hyperparameter settings
- After correcting the tuning grid, GRU4Rec performance improved substantially
- Tuned GRU4Rec remained below the best VSTAN configuration on the current Yoochoose sample setup

---

## Best Observed Performance

Current observations indicate:

- VSTAN currently achieves the best overall `MRR@10`
- Session-based nearest-neighbor methods remain highly competitive against neural approaches
- Hyperparameter tuning has a significant impact on GRU4Rec performance

---

## Technical Improvements

- Improved result reporting and CSV export structure
- Reduced unnecessary NaN-heavy output in analysis tables
- Added reusable tuning analysis workflow for future experiments
- Added automatic extraction of best-performing configurations

### Generated Outputs Include

- overall best tuning configurations
- best configuration per model
- summarized CSV reports for later visualization and thesis integration

---

## Role of This Step

This step marks the transition from:

- initial model implementation

toward:

- systematic experimental evaluation
- comparative model optimization
- reproducible session-based benchmarking

---

## Next Steps

- integrate tuned configurations into the final comparison pipeline
- scale experiments to larger Yoochoose subsets
- perform longer and larger tuning runs
- compare runtime and recommendation quality across models
- prepare visualizations and evaluation tables for the thesis


## 2026-05-15 – Persistent experiment logging and measurement infrastructure

### Overview

* Extended the session-based tuning pipeline with persistent experiment logging
* Added automatic runtime measurement for all experiment executions
* Added structured experiment tracking for:
  * successful runs
  * failed runs
  * runtime information
  * serialized configuration storage
* Implemented reusable experiment logging utilities for long-running experiments and server execution
* Added preparation utilities for additional evaluation metrics beyond ranking accuracy

---

### Experiment Logging

A new experiment logging system was introduced to support:

* incremental CSV-based result persistence
* automatic saving after each completed run
* fault tolerance for long-running tuning jobs
* reproducible experiment tracking

Stored information now includes:

* model name
* dataset
* hyperparameter configuration
* runtime
* device (CPU/GPU)
* evaluation metrics
* execution status
* error messages for failed runs

This enables unattended large-scale experimentation on external servers.

---

### Measurement Infrastructure

Additional evaluation utilities were prepared for future integration, including:

* recommendation coverage
* average recommendation popularity
* popularity bias analysis

These metrics will complement ranking-based evaluation metrics such as:

* Hit@K
* NDCG@K
* MRR@K

---

### Motivation

The initial evaluation pipeline focused mainly on ranking accuracy.

The new infrastructure prepares the framework for:

* larger-scale experiments
* automated multi-dataset benchmarking
* long-running hyperparameter tuning
* reproducible evaluation workflows
* additional beyond-accuracy measurements

---

### Role of This Step

This step marks the transition from:

* manual experimental execution

toward:

* scalable automated experimentation
* server-based evaluation workflows
* reproducible experiment management

---

### Next Steps

* integrate coverage and popularity-based metrics into evaluation runs
* extend tuning to larger parameter grids
* execute longer GPU-based tuning runs
* scale experiments to larger Yoochoose subsets
* prepare multi-dataset evaluation pipelines for Adressa and Globo


## 2026-05-15 – Extended session tuning measurements with runtime and beyond-accuracy metrics

### Overview

* Extended the session-based tuning pipeline with additional evaluation measurements
* Added runtime tracking for:
  * total runtime
  * training runtime
  * evaluation runtime
  * additional metric computation runtime
* Added beyond-accuracy measurements:
  * Coverage@10
  * Average Recommendation Popularity@10
* Stored all additional measurements in both tuning result files and persistent experiment logs
* Verified that all tuning runs completed successfully using CUDA acceleration

---

### Purpose

The goal of this step was to move beyond pure ranking accuracy and prepare the evaluation pipeline for larger-scale and long-running experiments.

The new measurements allow future analysis of:

* recommendation quality
* runtime-performance tradeoffs
* catalog coverage
* popularity bias
* model scalability

---

### Observations

* VS-KNN and VSTAN achieve strong ranking performance but require more runtime due to nearest-neighbor search
* GRU4Rec runs substantially faster on GPU, but still remains below VSTAN in the current tuning setup
* Runtime tracking is essential for comparing neural and non-parametric session-based models fairly

---

### Role of This Step

This step expands the experimental infrastructure from simple metric reporting to a more complete evaluation setup.

It prepares the project for unattended server-based experiments across larger datasets and wider hyperparameter grids.



## 2026-05-15 – Improved analysis workflow for session tuning results

### Overview

* Updated the session tuning analysis script to include new measurement dimensions
* Extended best-configuration reports with:
  * Coverage@10
  * Average Recommendation Popularity@10
  * runtime measurements
  * device information
  * execution status
* Generated updated best-overall and best-per-model result tables
* Prepared analysis outputs for later use in thesis tables and visualizations

---

### Analysis Outputs

Generated outputs include:

* `best_session_tuning_configurations_overall.csv`
* `best_session_tuning_configurations_per_model.csv`

These files summarize the strongest configurations according to MRR@10 while preserving additional evaluation dimensions.

---

### Interpretation

The analysis workflow now supports both:

* accuracy-focused model selection
* broader evaluation of efficiency and recommendation behavior

This is important because the best model by ranking accuracy may not always be the best model in terms of runtime, coverage, or popularity bias.

---

### Next Steps

* run larger tuning grids using the extended logging pipeline
* prepare larger Yoochoose subsets
* compare tuned models across sample sizes
* integrate the same measurement structure into future datasets such as Globo and Adressa


## 2026-05-16 – Multi-dataset session-based evaluation on Yoochoose and Globo

### Overview

* Successfully executed the unified session-based tuning pipeline on multiple datasets
* Evaluated:
  * VS-KNN
  * VSTAN
  * GRU4Rec
* across:
  * Yoochoose sample
  * Globo sample

The experiments were executed using the shared RecBole-based tuning framework with automated logging and extended evaluation metrics.

---

### Evaluation Dimensions

The pipeline now supports automated evaluation of:

#### Ranking Metrics
* Hit@10
* NDCG@10
* MRR@10

#### Beyond-Accuracy Metrics
* Coverage@10
* Average Recommendation Popularity@10

#### Efficiency Metrics
* total runtime
* training runtime
* evaluation runtime
* additional metric runtime

---

### Observations

* VSTAN currently achieves the strongest ranking performance on Yoochoose
* VS-KNN remains highly competitive despite its simpler neighborhood-based structure
* GRU4Rec benefits strongly from GPU acceleration and significantly lower runtime
* Globo produces lower recommendation accuracy overall, indicating a more challenging and dynamic recommendation setting

---

### Importance for the Thesis

This step establishes a reproducible multi-dataset benchmarking environment for session-based recommendation experiments.

The infrastructure now supports:

* automated experimentation
* large-scale hyperparameter tuning
* cross-dataset comparison
* runtime and popularity-bias analysis

This forms the technical foundation for the later large-scale evaluation phase of the thesis.

---

### Next Steps

* integrate Adressa dataset
* create larger Yoochoose and Globo subsets
* expand hyperparameter grids
* prepare long-running server-side experiments
* generate comparative plots and analysis tables



# 2026-05-18 – Initial integration of the Adressa news dataset

## Overview

- Started the integration of the Adressa dataset into the session-based RecBole evaluation pipeline
- Downloaded and prepared the larger multi-week Adressa dataset for future large-scale experiments
- Analyzed the raw dataset structure and verified compatibility with the existing preprocessing infrastructure
- Implemented the first version of the Adressa preprocessing pipeline for RecBole

---

## Dataset Characteristics

Adressa represents a large-scale news recommendation dataset with highly dynamic user behavior and rapidly changing item popularity.

The raw dataset consists of multiple daily interaction files:

```text
20170101
20170102
20170103
...
```

Each file contains JSON-based interaction events stored line-by-line.

The dataset includes information such as:

- user identifiers
- article identifiers
- timestamps
- URLs
- article metadata
- keyword profiles
- publishing information

---

## Initial Preprocessing Strategy

For the first session-based experiments, the preprocessing pipeline focuses only on the interaction information required for recommendation experiments.

### Current Mapping

| Original Field | Mapped Field |
|---|---|
| userId | user_id |
| id | item_id |
| time | timestamp |

Additional metadata fields are currently ignored and may later be used for extended experiments.

---

## Technical Improvements

The preprocessing pipeline was designed to support large-scale datasets by:

- reading files line-by-line
- avoiding full-memory JSON loading
- filtering invalid events during parsing
- generating RecBole-compatible `.inter` files
- automatically creating smaller sample datasets for faster experimentation

### Generated Outputs

- `adressa_recbole.inter`
- `adressa_recbole_sample.inter`

---

## Importance for the Thesis

Adressa introduces a highly dynamic news recommendation scenario into the evaluation framework.

This is especially relevant for:

- time-aware recommendation analysis
- popularity drift evaluation
- session-based recommendation
- cross-domain comparison

The dataset complements the existing evaluation setup:

| Dataset | Domain |
|---|---|
| MovieLens | long-term movie preferences |
| Amazon | product recommendation |
| Yoochoose | e-commerce sessions |
| Globo | news recommendation |
| Adressa | highly dynamic news recommendation |

---

## Planned Next Steps

- finalize the Adressa preprocessing pipeline
- generate the first Adressa sample dataset
- execute initial RecBole experiments on Adressa
- integrate Adressa into the automated tuning framework
- compare session-based model behavior across multiple news datasets



# 2026-05-18 – Preparation of the full-scale session-based tuning pipeline

## Overview

Designed and implemented a dedicated full-scale tuning pipeline for long-running recommendation experiments.

Separated the lightweight sample-based tuning workflow from the large-scale evaluation workflow.

Added support for:

- full dataset evaluation
- large hyperparameter grids
- automatic resume functionality
- completed-run skipping
- persistent result storage
- long-running experiment execution

The new infrastructure is intended for multi-day or week-long experimental runs on high-performance hardware.

---

## Motivation for a Separate Full Tuning Pipeline

The existing tuning pipeline was primarily designed for:

- debugging
- rapid iteration
- preprocessing validation
- quick metric verification

using smaller dataset samples.

As the project evolved toward large-scale experimentation, a separate full tuning workflow became necessary in order to:

- avoid accidental modification of the lightweight test pipeline
- support stable long-running execution
- simplify server-side experiment management
- allow interruption and continuation of experiments
- improve reproducibility of large-scale evaluations

The project now distinguishes between:

### `tune_session_models.py`

Sample-based testing and debugging.

and

### `tune_session_models_full.py`

Large-scale experimental evaluation.

---

## Full Dataset Configuration

The full tuning pipeline currently targets:

- `yoochoose_recbole`
- `globo_recbole`
- `adressa_recbole`

These datasets represent different recommendation domains:

- e-commerce recommendation
- news recommendation
- highly dynamic temporal recommendation settings

---

## Hyperparameter Grid Design

The selected hyperparameter grids were intentionally designed as a balanced compromise between:

- experimental diversity
- computational feasibility
- runtime limitations
- recommendation quality exploration

The goal is to generate sufficiently diverse configurations while still remaining executable within approximately one week on modern high-performance hardware.

---

## Selected Hyperparameter Grids

### VS-KNN

```python
k = [100, 200, 500]
sample_size = [500, 1000]
```

#### Rationale

- explores different neighborhood sizes
- evaluates local vs. broader session similarity
- balances recommendation quality and runtime complexity

---

### VSTAN

```python
k = [100, 200, 500]
sample_size = [500, 1000]
position_decay = [0.05, 0.1, 0.2]
idf_weighting = [True, False]
```

#### Rationale

- evaluates different temporal weighting strengths
- compares popularity-aware vs. IDF-based weighting
- explores trade-offs between short-term and broader session influence

---

### GRU4Rec

```python
hidden_size = [128, 256]
learning_rate = [0.001, 0.0005, 0.0001]
dropout_prob = [0.1, 0.2]
epochs = [10, 20]
```

#### Rationale

- explores different neural model capacities
- evaluates training stability under different learning rates
- analyzes regularization effects through dropout
- compares shorter vs. longer training durations

---

## Infrastructure Improvements

The full tuning pipeline additionally supports:

- automatic experiment resumption
- completed-run detection
- fault-tolerant execution
- runtime tracking
- experiment logging
- configuration serialization
- beyond-accuracy metrics

All experiment results are stored incrementally after each completed run to reduce the risk of data loss during long-running experiments.

---

## Expected Runtime Characteristics

The current tuning configuration is expected to generate:

- several hundred experiment runs
- multi-day runtime behavior
- large-scale result logs
- extensive cross-dataset evaluation data

The tuning setup is specifically intended for execution on GPU-enabled high-performance systems.

---

## Importance for the Thesis

This marks the transition from:

### implementation-focused development

toward:

### large-scale experimental evaluation
### cross-dataset benchmarking
### reproducible recommendation system analysis

The resulting infrastructure forms the foundation for the empirical evaluation phase of the thesis.

---

## Planned Next Steps

- validate the full tuning pipeline on small smoke-test runs
- execute large-scale multi-day experiments
- analyze runtime scalability across datasets
- generate comparative plots and result tables
- investigate popularity bias and coverage behavior across models
- compare simple neighborhood-based models against neural recommenders on large-scale datasets




## 2026-05-18 – Initial session-based experiments on Adressa

### Overview

* Successfully executed the first session-based recommendation experiments on the Adressa dataset
* Evaluated:
  * VS-KNN
  * VSTAN
  * GRU4Rec
* using the RecBole-based session tuning infrastructure

---

### Initial Observations

The first Adressa experiments produced noticeably different model behavior compared to Yoochoose.

Current ranking performance:

* GRU4Rec achieved the strongest recommendation accuracy
* VSTAN remained competitive
* VS-KNN showed lower performance on the highly dynamic news recommendation setting

---

### Runtime Observations

A major runtime difference between neural and neighborhood-based approaches was observed.

* GRU4Rec executed significantly faster due to GPU acceleration
* VS-KNN and VSTAN required substantially longer runtime because of large-scale session similarity computations

---

### Scientific Relevance

The first Adressa results indicate that:

* dataset characteristics strongly influence recommender performance
* neural sequential models may benefit from highly dynamic news environments
* recommendation quality and runtime scalability differ significantly across domains

These observations directly support the thesis focus on:

* fair recommender evaluation
* cross-domain comparison
* temporal recommendation analysis
* neural vs. non-neural recommendation behavior

---

### Next Steps

* integrate Adressa into the full-scale tuning pipeline
* execute larger multi-dataset experiments
* analyze runtime scalability across datasets
* compare popularity bias and coverage behavior
* generate comparative plots and evaluation tables


## 2026-05-19 – Multi-dataset session tuning results on Yoochoose, Globo, and Adressa

### Overview

* Executed the session-based tuning pipeline on three sample datasets:
  * Yoochoose sample
  * Globo sample
  * Adressa sample
* Evaluated the implemented session-based recommenders:
  * VS-KNN
  * VSTAN
  * GRU4Rec
* Used the unified RecBole-based tuning infrastructure with:
  * ranking metrics
  * beyond-accuracy metrics
  * runtime measurements
  * persistent experiment logging
  * CUDA acceleration

---

### Evaluation Setup

The experiments used the sample-based tuning pipeline with the following datasets:

* `yoochoose_recbole_sample`
* `globo_recbole_sample`
* `adressa_recbole_sample`

The evaluation included:

* Hit@10
* NDCG@10
* MRR@10
* Coverage@10
* Average Recommendation Popularity@10
* training runtime
* evaluation runtime
* total runtime

---

### Main Results

#### Yoochoose

The best observed Yoochoose results were achieved by VSTAN:

* Hit@10: approximately 0.514
* NDCG@10: approximately 0.328
* MRR@10: approximately 0.270

VS-KNN remained highly competitive, while GRU4Rec achieved competitive results with substantially lower runtime.

---

#### Adressa

The best observed Adressa results were achieved by GRU4Rec:

* Hit@10: approximately 0.518
* Runtime: around one minute for the tested configuration

This differs from Yoochoose, where the nearest-neighbor models achieved the strongest ranking results.

---

#### Globo

Globo was successfully included in the multi-dataset tuning pipeline.

Although Globo did not appear among the global Top-10 configurations by MRR@10 in the current run, it is now fully supported by the same automated evaluation infrastructure.

---

### Observations

* The strongest model differs by dataset:
  * VSTAN performs best on Yoochoose
  * GRU4Rec performs best on Adressa
* This confirms that recommendation performance is highly dataset-dependent.
* Neighborhood-based session models remain strong and competitive.
* Neural sequential models can be especially effective on dynamic news recommendation data.
* GRU4Rec benefits strongly from GPU acceleration and shows substantially lower runtime.
* VS-KNN and VSTAN require more runtime due to session-neighborhood computations.

---

### Interpretation

The results support an important thesis argument:

There is no universally best recommender model. Instead, model performance depends strongly on:

* dataset characteristics
* temporal dynamics
* session structure
* item popularity distribution
* scalability constraints

The results also highlight the importance of evaluating both:

* recommendation accuracy
* computational efficiency

---

### Role of This Step

This step establishes the first complete multi-dataset session-based evaluation setup.

It shows that the implemented framework can now compare different model families across different domains using the same evaluation and logging infrastructure.

---

### Next Steps

* use the analysis outputs to compare best configurations per dataset and model
* prepare full-scale tuning runs using `tune_session_models_full.py`
* execute longer server-based experiments on full datasets
* further analyze coverage and popularity-bias behavior
* generate thesis-ready result tables and plots


## 2026-05-27 - Updated full tuning grids for session-based and Top-N models

### Overview

Updated the large-scale full tuning configuration for both session-based recommendation models and Top-N recommendation models.

Changed files:

- `src/recbole_framework/tuning/tune_session_models_full.py`
- `src/recbole_framework/tuning/tune_topn_models_full.py`

---

### Session-Based Recommendation Grid Updates

#### VS-KNN

Updated the grid to:

```python
vsknn_k = [100, 200, 500]
vsknn_sample_size = [100, 250, 500]
```

This expands the search over different neighborhood sizes and candidate-session sample sizes.
The goal is to compare focused local neighborhoods against broader neighbor sets and to measure how larger candidate pools influence recommendation quality and runtime.

#### VSTAN

Updated the grid to:

```python
vstan_k = [100, 200, 500]
vstan_sample_size = [100, 250, 500]
vstan_position_decay = [0.05, 0.1, 0.2]
vstan_idf_weighting = [True, False]
```

This extends VS-KNN with position-aware weighting and optional IDF-based popularity normalization.
The grid makes it possible to analyze whether stronger focus on recent session positions improves ranking quality and whether reducing the influence of very popular items improves diversity and mitigates popularity bias.

#### GRU4Rec

Updated the grid to:

```python
hidden_size = [100, 128, 256]
learning_rate = [0.001, 0.0005, 0.0001]
dropout_prob = [0.1, 0.2]
epochs = [10, 20]
```

This broadens the neural sequential model search across model capacity, learning stability, regularization strength, and training duration.
The chosen grid supports analysis of whether GRU4Rec benefits from larger hidden representations and longer training, while still keeping the number of runs manageable for server execution.

---

### Top-N Recommendation Grid Updates

#### MostPop

MostPop remains without hyperparameters.
It continues to serve as the global popularity baseline and minimal reference point for comparing time-aware popularity models and personalized models.

#### RecentPop

Updated the grid to:

```python
window_days = [1, 3, 7, 14, 30, 60, 90, 180]
```

The added 180-day window allows comparison between short-term trend sensitivity and longer-term popularity stability.
This is important for contrasting highly dynamic domains such as news with more stable domains such as movies or product recommendation.

#### DecayPop

Updated the grid to:

```python
decay_lambda = [
    1e-9,
    5e-9,
    1e-8,
    5e-8,
    1e-7,
    5e-7,
    1e-6,
]
```

The expanded lambda range tests both weak and strong exponential time decay.
Small lambda values preserve older interactions for longer, while larger values emphasize recent interactions more strongly.

#### BPR

Updated the grid to:

```python
embedding_size = [32, 64, 128]
learning_rate = [0.001, 0.0005, 0.0001]
epochs = [50]
```

The number of epochs was fixed at 50 to reduce unnecessary runs and keep the full tuning setup computationally efficient.
The grid still explores the most important BPR dimensions: latent factor size and learning rate.

### Scientific Motivation

The updated grids support a stronger experimental design.
The goal is not only to find the single best configuration, but also to analyze which hyperparameter regions work well for different model classes and dataset domains.

This enables more meaningful thesis analysis across:

- session-based neighborhood methods
- neural sequential recommendation
- static popularity
- recent-window popularity
- exponential time-decayed popularity
- personalized matrix factorization

Expected insights include:

- VS-KNN and VSTAN should be strong on session datasets.
- Position weighting and IDF weighting may improve recommendation quality and reduce popularity bias.
- RecentPop and DecayPop should be especially relevant in news and trend-heavy domains.
- Larger time windows and weaker decay may be better for long-term domains.
- GRU4Rec may scale well with GPU acceleration, but is not guaranteed to outperform simpler methods.

Overall, this tuning setup is better suited for fair cross-model and cross-dataset evaluation than a narrow accuracy-only search.


## 2026-05-27 - Improved tuning reliability and result analysis workflow

### Overview

After reviewing the tuning and evaluation pipeline, several infrastructure-level improvements were implemented.
These changes do not alter the scientific model logic or the selected hyperparameter grids.
Instead, they make long-running experiments more reproducible, easier to resume, and safer to analyze.

Changed files:

- `src/recbole_framework/custom_models/session/vsknn_recbole.py`
- `src/recbole_framework/custom_models/session/vstan_recbole.py`
- `src/recbole_framework/custom_models/topn/mostpop_recbole.py`
- `src/recbole_framework/tuning/tune_topn_models_full.py`
- `src/recbole_framework/analysis/analyze_topn_tuning_results.py`
- `src/recbole_framework/analysis/analyze_session_tuning_results.py`
- `.gitignore`

---

### Deterministic Candidate Sampling for VS-KNN and VSTAN

VS-KNN and VSTAN use candidate-session sampling when the number of possible neighbor sessions is larger than the configured sample size.
Previously, this sampling used Python's global `random.sample`.

This was changed to use a deterministic local random generator based on:

- the configured experiment seed
- the current session item set

#### Why this was done

The goal is to improve reproducibility.
For thesis experiments, repeated runs with the same seed and configuration should evaluate the same sampled candidate sessions.
This makes comparison across hyperparameter settings more reliable and avoids small random differences caused by uncontrolled sampling.

---

### Removed Debug Output from MostPop

Temporary debug print statements were removed from the RecBole-native MostPop model.

#### Why this was done

MostPop is used as a baseline in repeated tuning and evaluation runs.
Debug output during large-scale experiments makes logs harder to read and can create unnecessary noise during server execution.
Removing these prints does not change the model scores or evaluation behavior.

---

### Robust Resume Support for Top-N Full Tuning

The Top-N full tuning script was updated to follow the same fault-tolerant structure already used in the session-based full tuning pipeline.

Added:

- stable `run_id` generation
- detection of already completed successful runs
- skipping of completed configurations
- incremental result persistence after each run
- failed-run persistence in the result CSV

#### Why this was done

Full tuning can run for many hours or days.
If a run is interrupted, the script should continue from already completed configurations instead of starting from scratch.
This is especially important for server-based experiments and prevents unnecessary repeated computation.

---

### Full-Tuning Analysis File Alignment

The tuning analysis scripts were updated to read the full tuning result files:

``` python
  topn_full_tuning_results.csv
  session_full_tuning_results.csv
```

instead of the smaller non-full tuning result files.

#### Why this was done

The expanded grids write their results to the full tuning output files.
The analysis scripts must read those same files so that best-configuration reports reflect the current full tuning experiments rather than older sample or debugging runs.

---

### Added Git Ignore Rules for Local Artifacts

A `.gitignore` file was added to exclude local and generated artifacts such as:

- virtual environments
- raw and processed datasets
- RecBole-formatted datasets
- experiment results
- TensorBoard logs
- model checkpoints
- Python cache files
- IDE-local metadata

#### Why this was done

The workspace contains many large generated files and local machine artifacts.
These should not be accidentally committed because they make the repository hard to maintain and can mix local experiment state with reproducible source code.

---

### Methodological Note on Data Splitting

During the review, one remaining methodological issue was identified but not changed automatically:

The current Top-N RecBole preparation is not fully consistent across datasets.
MovieLens is prepared from `movielens_train.csv`, while Amazon is prepared from `amazon_interactions.csv`.
Both are then split again internally by RecBole using the configured temporal split.

For the final thesis evaluation, the recommended approach is to use one consistent strategy:

- prepare both MovieLens and Amazon from the full interaction files
- let RecBole apply the same time-ordered train/validation/test split to both datasets
- document this as the unified evaluation protocol

This would better match the thesis goal of fair cross-model and cross-dataset comparison.

The change was not applied in this step because it changes the evaluation protocol and should be treated as a deliberate experimental-design decision.


## 2026-05-27 - Unified Top-N data preparation and sample-to-full session evaluation

### Overview

The experimental workflow was updated to separate hyperparameter search from final full-dataset evaluation more clearly.

Changed files:

- `src/recbole_framework/datasets/topn/prepare_recbole_movielens.py`
- `src/recbole_framework/tuning/evaluate_session_models_final.py`
- `src/recbole_framework/tuning/run_all_full_tuning.py`

---

### Unified Top-N RecBole Data Preparation

MovieLens RecBole preparation was changed from:

``` python
 movielens_train.csv
```

to:

``` python
 movielens_interactions.csv
```

Amazon already uses the full interaction file.
With this change, both Top-N datasets are prepared from their complete interaction histories and are then split by RecBole using the same configured time-ordered evaluation protocol.

#### Why this was done

The previous setup mixed two strategies:

- MovieLens was prepared from a pre-split training file.
- Amazon was prepared from the full interaction file.
- Both were then split again internally by RecBole.

This could make cross-dataset comparisons harder to interpret because the evaluation protocol was not aligned.
Preparing both datasets from full interactions makes the Top-N setup methodologically cleaner and better suited for fair comparison between MostPop, RecentPop, DecayPop, and BPR.

---

### Session-Based Final Full-Dataset Evaluation

Added a dedicated final evaluation script:

``` python
evaluate_session_models_final.py
```

This script:

- reads the sample-based session tuning results
- selects the top configurations per dataset and model
- maps sample datasets to their full dataset versions
- evaluates only these selected configurations on the full datasets
- stores results separately in `session_final_full_evaluation_results.csv`

The sample-to-full mapping is:

``` python
yoochoose_recbole_sample -> yoochoose_recbole
globo_recbole_sample -> globo_recbole
adressa_recbole_sample -> adressa_recbole
```

#### Why this was done

Running the complete VS-KNN/VSTAN/GRU4Rec grid on full session datasets is computationally expensive.
Neighborhood-based models such as VS-KNN and VSTAN are especially costly because they search over candidate sessions during prediction.

The new workflow follows a more scalable and scientifically defensible design:

1. run broad hyperparameter tuning on representative samples
2. identify the best configurations per model and dataset
3. run only those selected configurations on the full datasets

This keeps the experimental setup feasible while still allowing final conclusions to be based on full-dataset evaluation.

---

### Updated Full Tuning Runner

`run_all_full_tuning.py` now runs:

1. sample-based session full-grid tuning
2. selected session full-dataset final evaluation
3. Top-N full-grid tuning

#### Why this was done

The runner now reflects the intended experimental workflow:

- sample grids for expensive session-model tuning
- full-data final validation for selected session configurations
- full-data Top-N tuning where the datasets and models are computationally more manageable

---

### Scientific Rationale

Using samples for hyperparameter search is scientifically acceptable when it is clearly documented and followed by full-dataset validation.

This distinction is important:

- sample results are used for model selection and hyperparameter exploration
- full results are used for final reporting and cross-dataset comparison

This makes the project more computationally realistic without weakening the evaluation design.


## 2026-06-09 - Time-based session sampling and session popularity baselines

### Overview

The session-based experimental setup was extended in two ways:

- sample-size handling was changed to prefer the newest temporal data
- popularity baselines were added to the session-based RecBole tuning pipeline

Changed files:

- `src/recbole_framework/datasets/session/prepare_yoochoose_recbole_sample.py`
- `src/recbole_framework/datasets/session/prepare_globo_recbole.py`
- `src/recbole_framework/datasets/session/prepare_adressa_recbole.py`
- `src/recbole_framework/custom_models/session/vsknn_recbole.py`
- `src/recbole_framework/custom_models/session/vstan_recbole.py`
- `src/recbole_framework/custom_models/session/popularity_recbole.py`
- `src/recbole_framework/tuning/tune_session_models.py`
- `src/recbole_framework/tuning/tune_session_models_full.py`
- `src/recbole_framework/tuning/evaluate_session_models_final.py`
- `src/recbole_framework/analysis/analyze_session_tuning_results.py`

---

### Time-Based Sample Selection

Session dataset sample creation was changed from random or oldest-first sampling to newest-first sampling.

For Yoochoose and Globo, sessions are now ranked by their latest interaction timestamp.
The sample is then filled with the newest sessions until the configured interaction budget is reached.
This keeps sessions intact while making the sample temporally recent.

For Adressa, sample creation was changed from:

``` python
recbole_df.head(sample_size)
```

to:

``` python
recbole_df.tail(sample_size)
```

Because Adressa interactions are sorted by timestamp before conversion, this selects the newest interactions instead of the oldest ones.

#### Why this was done

The project evaluates time-aware popularity and session-based recommendation models.
Using old or random samples can weaken temporal interpretability because the sample may not represent the latest interaction behavior.

The updated sampling strategy makes sample datasets more appropriate for experiments that depend on recency.

---

### Newest Candidate Sessions for VS-KNN and VSTAN

The `sample_size` parameters in the session-neighborhood models were changed:

- `vsknn_sample_size`
- `vstan_sample_size`

Previously, when the candidate session set was larger than the configured sample size, candidates were selected randomly with a deterministic seed.
Now, candidate sessions are sorted by their reference timestamp and the newest candidates are selected.

This means values such as `100`, `250`, `500`, or `1000` now represent the most recent candidate sessions rather than random candidate sessions.

#### Why this was done

VS-KNN and VSTAN are session-neighborhood methods.
Their candidate sampling directly affects which historical sessions can influence the recommendation.

Selecting the newest candidates better matches the time-aware goal of the project and makes the `sample_size` parameter easier to interpret experimentally.

---

### Session-Based Popularity Baselines

Added session-compatible popularity models:

- `SessionMostPopRecBole`
- `SessionRecentPopRecBole`
- `SessionDecayPopRecBole`

These models inherit from RecBole's sequential recommender interface so they can run inside the same session-based evaluation setup as:

- VS-KNN
- VSTAN
- GRU4Rec

The models act as global popularity baselines under a session-based protocol:

- MostPop ranks items by overall interaction count
- RecentPop ranks items by interaction count within a recent time window
- DecayPop ranks items with exponential time-decayed interaction weights

#### Why this was done

This makes it possible to compare simple popularity-based methods against session-aware models under the same data split, metrics, and sample datasets.
It helps answer whether time-aware popularity remains competitive in a session-based environment.

---

### Session Grid Search Extensions

The session tuning scripts now include MostPop, RecentPop, and DecayPop.

Small session tuning uses:

``` python
RecentPop window_days = [7, 30]
DecayPop decay_lambda = [1e-7, 1e-6]
```

Full session tuning uses:

``` python
RecentPop window_days = [1, 3, 7, 14, 30, 60, 90, 180]
DecayPop decay_lambda = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
```

MostPop is run once per session sample dataset as a non-parametric baseline.

The analysis and final evaluation scripts were also extended so that:

- `window_days` and `decay_lambda` remain visible in session result summaries
- the best sample-based popularity configurations can be evaluated on the full session datasets

---

### Validation

The modified Python files were checked with `py_compile`.
The syntax check passed.

A direct import check with the bundled Codex Python runtime was not possible because that runtime does not include `torch`.
The actual project environment with RecBole and PyTorch is still required for execution.


## 2026-06-09 - Popularity weighting for session-neighborhood scoring

### Overview

VS-KNN and VSTAN were extended with an optional popularity-weighting feature inspired by the RSC18 session-KNN implementation.

Changed files:

- `src/recbole_framework/custom_models/session/vsknn_recbole.py`
- `src/recbole_framework/custom_models/session/vstan_recbole.py`
- `src/recbole_framework/tuning/tune_session_models.py`
- `src/recbole_framework/tuning/tune_session_models_full.py`
- `src/recbole_framework/tuning/evaluate_session_models_final.py`
- `src/recbole_framework/analysis/analyze_session_tuning_results.py`
- `src/recbole_framework/analysis/evaluate_recbole_results.py`

Reference implementation:

- `https://github.com/rn5l/rsc18/blob/master/algorithms/knn/sknn.py`

---

### Scoring Change

The new feature follows the same general idea as the `pop_weight` option in the referenced session-KNN code:
candidate item scores can be divided by item popularity during scoring.

The implementation uses a numeric exponent instead of a boolean flag:

``` python
weighted_score = score / (item_popularity ** popularity_weight)
```

This makes the feature tunable:

- `0.0` disables popularity weighting and preserves the previous behavior
- `0.5` applies a softer inverse-popularity correction
- `1.0` applies the full inverse-popularity correction

Item popularity is computed once during model initialization from the training interactions available in RecBole's internal dataset.
The feature does not require a separate sampling step.

---

### New Hyperparameters

Added to VS-KNN:

``` python
vsknn_popularity_weight
```

Added to VSTAN:

``` python
vstan_popularity_weight
```

The small session tuning script now tests:

``` python
[0.0, 1.0]
```

The full session tuning script now tests:

``` python
[0.0, 0.5, 1.0]
```

This keeps the old behavior in the grid while adding explicit popularity-weighted variants.

---

### Evaluation and Runtime Tracking

The new hyperparameters are included in:

- session tuning result CSVs
- session tuning analysis tables
- final full-dataset session evaluation
- structured RecBole report configuration summaries

No new runtime-measurement mechanism was needed because the existing session tuning pipeline already records:

- `runtime_seconds`
- `train_runtime_seconds`
- `eval_runtime_seconds`
- `extra_metrics_runtime_seconds`

Every popularity-weighted configuration therefore receives the same runtime tracking as the existing VS-KNN and VSTAN configurations.


## 2026-06-24 - Refined session popularity baselines

### Motivation

The first session-based popularity evaluation showed that `MostPop`, `RecentPop`, and `DecayPop` were often identical or nearly identical in the best-result tables.

This was not caused by a broken scorer. The issue was the temporal parameterization:

- large fixed `RecentPop` windows often covered nearly the whole sampled dataset
- very small `DecayPop` lambdas produced almost no temporal decay
- both effects can make the time-aware popularity baselines collapse back into global `MostPop`

Observed sample time spans:

| Dataset | Approximate time span |
|---|---:|
| `adressa_recbole_sample` | 1.4 days |
| `globo_recbole_sample` | 30.8 days |
| `yoochoose_recbole_sample` | 182.0 days |

Because these spans differ strongly, fixed day windows are difficult to compare across datasets.

---

### Implementation Changes

Changed files:

- `src/recbole_framework/custom_models/session/popularity_recbole.py`
- `src/recbole_framework/tuning/tune_session_popularity_refined.py`
- `src/recbole_framework/tuning/evaluate_session_models_final.py`
- `src/recbole_framework/analysis/evaluate_recbole_results.py`
- `src/recbole_framework/analysis/analyze_session_tuning_results.py`

`MostPop` remains unchanged and continues to act as the global popularity anchor.

`RecentPop` now supports an additional relative time-window parameter:

``` python
recent_fraction
```

This selects the most recent fraction of the training time span. For example:

``` text
recent_fraction = 0.25
```

means that only interactions in the most recent 25% of the observed training time span are counted.

The old fixed-day parameter remains supported:

``` python
window_days
```

`DecayPop` now supports an additional half-life parameter:

``` python
decay_half_life_days
```

This is converted internally into an exponential decay lambda:

``` python
decay_lambda = log(2) / half_life_seconds
```

This makes the decay easier to explain because a half-life of `1` day means that an interaction loses half of its weight after one day.

The old direct lambda parameter remains supported:

``` python
decay_lambda
```

---

### Isolated Refined Popularity Run

A new isolated tuning script was added:

``` text
src/recbole_framework/tuning/tune_session_popularity_refined.py
```

It writes separate outputs:

``` text
recbole_results/tuning_results/session_popularity_refined_results.csv
recbole_results/experiment_logs/session_popularity_refined_log.csv
```

The refined grid tests:

``` python
RecentPop recent_fraction = [0.01, 0.05, 0.10, 0.25, 0.50]
DecayPop decay_half_life_days = [0.25, 0.5, 1, 3, 7, 14, 30]
```

`MostPop` is run once per dataset.

Total planned runs:

``` text
3 datasets * (1 MostPop + 5 RecentPop + 7 DecayPop) = 39 runs
```

The first DecayPop attempt failed because RecBole returned `None` for an unset optional config value. The config helper was updated to treat `None` like a missing value. The successful rerun produced all 39 successful refined runs. The result CSV still contains the 21 earlier failed rows, so evaluation should filter:

``` python
status == "success"
```

---

### Best Refined Popularity Results

Best successful refined configurations by `MRR@10`:

| Dataset | Model | Best refined config | Hit@10 | NDCG@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| `adressa_recbole_sample` | `DecayPop` | `decay_half_life_days = 0.25` | 0.4046 | 0.2231 | 0.1682 |
| `adressa_recbole_sample` | `RecentPop` | `recent_fraction = 0.25` | 0.3830 | 0.2131 | 0.1619 |
| `adressa_recbole_sample` | `MostPop` | global popularity | 0.3734 | 0.1841 | 0.1268 |
| `globo_recbole_sample` | `DecayPop` | `decay_half_life_days = 30` | 0.0619 | 0.0316 | 0.0226 |
| `globo_recbole_sample` | `MostPop` | global popularity | 0.0626 | 0.0316 | 0.0224 |
| `globo_recbole_sample` | `RecentPop` | `recent_fraction = 0.50` | 0.0212 | 0.0110 | 0.0080 |
| `yoochoose_recbole_sample` | `MostPop` | global popularity | 0.0215 | 0.0112 | 0.0080 |
| `yoochoose_recbole_sample` | `RecentPop` | `recent_fraction = 0.25` | 0.0166 | 0.0094 | 0.0072 |
| `yoochoose_recbole_sample` | `DecayPop` | `decay_half_life_days = 30` | 0.0166 | 0.0091 | 0.0069 |

---

### Findings

The refined baselines no longer all collapse into identical `MostPop` results.

Main findings:

- `Adressa` benefits strongly from refined time-aware popularity.
- `DecayPop` with a short half-life is the best popularity-only session baseline on `Adressa`.
- `RecentPop` with a relative window also improves clearly on `Adressa`.
- `Globo` does not benefit meaningfully from global time-aware popularity. `MostPop` and `DecayPop` are almost tied, while `RecentPop` is weaker.
- `Yoochoose` also does not benefit from the refined time-aware popularity variants. `MostPop` remains the strongest popularity-only baseline.

Interpretation:

``` text
Time-aware popularity is dataset-dependent. It can substantially improve a popularity-only baseline on Adressa, but it does not replace session-aware modeling on Globo or Yoochoose.
```

The refined setup is methodologically stronger than the original fixed-window/lambda setup because it avoids unintentionally comparing several near-`MostPop` variants and uses parameters that are easier to explain in the thesis.

---

## 2026-07-06 - Docker Server Preparation

### Goal

Prepare the project so the full RecBole experiments can be executed reproducibly on an external GPU server.

The server run should cover:

- Top-N models:
  - `MostPop`
  - `RecentPop`
  - `DecayPop`
  - optional `BPR`
- Session-based models:
  - `MostPop`
  - `RecentPop`
  - `DecayPop`
  - `VS-KNN`
  - `VSTAN`
  - `GRU4Rec`
- Full prepared RecBole datasets:
  - `movielens_recbole`
  - `amazon_recbole`
  - `adressa_recbole`
  - `globo_recbole`
  - `yoochoose_recbole`

### Docker Setup

A Docker-based server setup was added:

``` text
Dockerfile
docker/requirements-server.txt
.dockerignore
```

The Docker image is based on an NVIDIA CUDA image and installs the project dependencies needed for the RecBole experiments.

The datasets are intentionally not copied into the image. They are mounted at runtime so the same image can be used on different servers and with different dataset/result locations.

### Server Runner

A new server entry point was added:

``` text
src/recbole_framework/tuning/run_server_full_experiments.py
```

It combines the full Top-N and session-based tuning runs into one resumable server script.

Default Top-N outputs:

``` text
recbole_results/tuning_results/server_full_topn_results.csv
recbole_results/experiment_logs/server_full_topn_log.csv
```

Default session outputs:

``` text
recbole_results/tuning_results/server_full_session_results.csv
recbole_results/experiment_logs/server_full_session_log.csv
```

The output filenames use the `server_full` prefix so local experiment files are not overwritten.

### Default Server Grid

Top-N default grid:

- `MostPop`: 1 run per dataset
- `RecentPop`: 8 time windows per dataset
- `DecayPop`: 7 decay lambdas per dataset
- default total: 32 Top-N runs across MovieLens and Amazon

Session default grid:

- `MostPop`: 1 run per dataset
- `RecentPop`: 5 relative recent fractions per dataset
- `DecayPop`: 7 half-life values per dataset
- `VS-KNN`: 36 configurations per dataset
- `VSTAN`: 216 configurations per dataset
- `GRU4Rec`: 36 configurations per dataset
- default total: 903 session runs across Adressa, Globo, and Yoochoose

Total default server run without BPR:

``` text
935 runs
```

### Data Mount Structure

The server runbook now documents that the server data folder must be mounted so the container sees:

``` text
/app/data/recbole/adressa_recbole/
/app/data/recbole/globo_recbole/
/app/data/recbole/yoochoose_recbole/
/app/data/recbole/movielens_recbole/
/app/data/recbole/amazon_recbole/
```

Each dataset folder must contain the matching RecBole `.inter` file, for example:

``` text
globo_recbole/globo_recbole.inter
```

### Runbook Documentation

The server documentation was expanded:

``` text
docs/server_docker_run.md
```

It now includes:

- server requirements
- expected project location
- exact dataset folder structure
- exact result folder structure
- Docker build command
- smoke test command
- full run command
- Top-N-only command
- session-only command
- selected model/dataset examples
- resume behavior after interruption
- monitoring commands
- common error cases
- final files to collect after the server run

### Resume Behavior

The server runner reuses the existing `run_id` based resume logic.

If the run is stopped and restarted with the same `--output-prefix`, already completed successful runs are skipped.

This is important for long-running full-dataset experiments, especially for `VSTAN` on session data.

### Persistent Server Execution

The server runbook was extended with a dedicated `tmux` section.

The intended server workflow is:

``` text
tmux new -s recbole-full
docker run --rm --gpus all ...
Ctrl+B, then d
tmux attach -t recbole-full
```

This ensures the Docker experiment keeps running even if the SSH connection is closed.

The documentation explicitly warns that `Ctrl+C` should only be used when the run should really be stopped.

### Validation

The new server runner was syntax-checked locally with:

``` text
python -m py_compile src/recbole_framework/tuning/run_server_full_experiments.py
```

Docker build and GPU execution were not run locally because they require the target server environment.

---

## 2026-07-06 - Reproducibility Documentation

### Goal

Document how the project results can be reproduced in a scientifically traceable way.

### Added Documentation

A new reproducibility guide was added:

``` text
docs/reproducibility.md
```

It describes:

- recording the exact Git branch and commit hash
- choosing Docker/server reproduction or a local Python environment
- expected RecBole dataset folder structure
- dataset provenance information to record
- RecBole evaluation settings
- primary and supporting metrics
- Docker server execution with `tmux`
- server output files
- local full-tuning output files
- how server output files relate to the existing structured evaluator
- how to generate the structured evaluation report
- which artifacts should be archived for a reproducible result package
- which information should be reported in the thesis
- expected limitations, especially for runtime comparability

The README now links to:

``` text
docs/reproducibility.md
docs/server_docker_run.md
```

This separates the short project overview from the detailed scientific reproduction workflow.

### README Synchronization

The README was synchronized with the current Docker/server state.

It now explicitly distinguishes between:

- the currently checked-in structured report based on local full-tuning files and session sample datasets
- the new Docker/server reproduction workflow, which uses the full session dataset names by default

The README also links the server runner and the Docker/reproducibility documentation from the script overview.

---

## 2026-07-17 - VSKNN Reference Audit and Cross-Domain RecBole Validation

### Goal

Prepare the existing VSKNN model for a possible contribution to the official
RecBole framework and verify that its implementation and evaluation are
scientifically defensible.

### Why This Work Was Necessary

The previous implementation used unweighted binary cosine similarity and
created one neighbor entry for every RecBole prefix-target row. It therefore
behaved more like plain SKNN than the published vector-multiplication VSKNN and
duplicated longer training sessions in the neighbor index.

RecBole's `SequentialDataset` creates multiple prefix-target examples from one
original session before splitting. The model was already initialized from
`train_data.dataset`, so no direct validation/test leakage was detected.
However, the augmented training rows had to be collapsed into one longest
available training session per session ID to avoid systematic over-weighting.

### Implementation Changes

- renamed the primary model class from `VSKNNRecBole` to `VSKNN`;
- retained `VSKNNRecBole` as a temporary compatibility alias;
- introduced the preferred parameters `neighbor_size`, `sample_size`,
  `sampling`, `similarity`, `session_weighting`, and `score_weighting`;
- implemented the reference `vec` similarity and optional weighted cosine;
- added position weights for the current session;
- added neighbor-score decay based on the most recent shared click;
- reconstructed exactly one reference session per training session ID;
- removed the thesis-specific popularity correction from the upstream-faithful
  VSKNN path;
- separated the pure algorithm into `vsknn_core.py` for direct unit testing;
- added deterministic score, weighting, reconstruction, and leakage tests.

### Multi-Dataset Runner

`run_vsknn_recbole.py` now accepts all sample and full session dataset names.
Examples:

```powershell
python -m src.recbole_framework.runners.session.run_vsknn_recbole --dataset globo_recbole_sample
python -m src.recbole_framework.runners.session.run_vsknn_recbole --all-samples
```

`--all-samples` evaluates Yoochoose, Globo, and Adressa sequentially and writes
both separate result files and a combined summary under:

```text
recbole_results/vsknn_audited/
```

### Fixed Smoke-Test Configuration

- RecBole: 1.2.1
- Python: 3.11.9
- PyTorch: 2.7.1 CPU
- seed: 42
- neighbor size: 100
- candidate sample size: 500
- candidate sampling: most recent sessions
- similarity: vector multiplication (`vec`)
- current-session weighting: division (`div`)
- neighbor-score weighting: division (`div`)
- split: ordered 80/10/10 ratio split

### Audited VSKNN Results

| Dataset | Hit@10 | NDCG@10 | MRR@10 | Reference sessions | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yoochoose sample | 0.5387 | 0.3470 | 0.2867 | 126,496 | 169.39 |
| Globo sample | 0.2916 | 0.1211 | 0.0697 | 175,715 | 218.84 |
| Adressa sample | 0.4276 | 0.2259 | 0.1650 | 98,032 | 2,002.70 |

### Comparison With the Stored Legacy Runs

The existing legacy rows with neighbor size 100 and sample size 500 report:

| Dataset | Legacy Hit@10 | Audited Hit@10 | Legacy NDCG@10 | Audited NDCG@10 | Legacy MRR@10 | Audited MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Yoochoose sample | 0.4947 | 0.5387 | 0.3177 | 0.3470 | 0.2624 | 0.2867 |
| Globo sample | 0.3373 | 0.2916 | 0.1326 | 0.1211 | 0.0713 | 0.0697 |
| Adressa sample | 0.3428 | 0.4276 | 0.1735 | 0.2259 | 0.1225 | 0.1650 |

The audited algorithm improves all reported quality metrics on Yoochoose and
Adressa but slightly reduces them on Globo. This domain-specific outcome is
important: the correction does not simply inflate every score, and conclusions
must be based on all datasets rather than Yoochoose alone.

The legacy runs were recorded on CUDA while the audited validation was run on
CPU. Their runtime numbers must therefore not be compared directly. Within the
new CPU run, Adressa is a substantial runtime outlier. This indicates that
candidate overlap and evaluation workload matter more than the raw number of
reference sessions alone.

### Validation

- 13 automated unit/CLI tests passed;
- RecBole constructed train/validation/test splits successfully;
- all three sample evaluations completed without model or leakage errors;
- repeated Yoochoose execution reproduced the same ranking metrics;
- result files were written separately to prevent accidental overwrites.

### Performance Profiling and Optimization

A deterministic profile was run on the first 2,000 Adressa test queries.

Before optimization:

- elapsed time: 43.321 seconds;
- throughput: 46.17 queries per second;
- 36.734 seconds were spent in candidate-session selection;
- global candidate sorting generated about 59 million key-function calls.

The bottleneck was the construction and complete recency sort of the union of
all sessions containing any current-session item. Only the newest 500 sessions
were subsequently used, so most sorting work was discarded.

The implementation was changed to:

- sort each item's session list by recency once during index construction;
- lazily merge relevant sorted lists with `heapq.merge`;
- stop after `sample_size` unique candidate sessions;
- prepare position weights once per query;
- prepare reverse item positions once per query for score decay.

After optimization, the identical 2,000 queries required 8.076 seconds or
247.65 queries per second. This is a 5.36x microbenchmark speedup.

Four candidate-order tests and two precomputation-equivalence tests were added.
All 19 unit and CLI tests passed. Full RecBole reruns produced exactly the same
Hit, NDCG, and MRR values as before optimization.

### Optimized End-to-End Runtime

| Dataset | Before (s) | After (s) | Reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Yoochoose sample | 169.39 | 134.90 | 20.4% | 1.26x |
| Globo sample | 218.84 | 121.94 | 44.3% | 1.79x |
| Adressa sample | 2,002.70 | 147.53 | 92.6% | 13.57x |

Adressa benefits most because its query candidate unions were especially large.
The optimized runtimes are now in a comparable range across all three samples.

### Next Steps

1. Update the VSKNN tuning grid to use the audited parameter names and weighting
   dimensions.
2. Run the revised grid on consistent server hardware.
3. Regenerate the structured evaluation report from audited results.
4. Open the prepared RecBole proposal issue and request maintainer guidance for
   the lifecycle of non-parametric models.

---

## 2026-07-17 - Compact Audited VSKNN Parameter Tuning

### Goal

Replace the obsolete tuning conclusions from the former SKNN-like
implementation with a small, interpretable tuning study for the audited VSKNN
algorithm. Only VSKNN had to be rerun; datasets, splits, seeds, GRU4Rec,
popularity baselines, BPR, and unchanged VSTAN results were not recomputed.

### Why a Compact Grid Was Used

The previous VSKNN grid used legacy parameter names and included the
thesis-specific popularity correction that is intentionally excluded from the
upstream-faithful implementation. A full Cartesian product of every weighting
choice would require many redundant runs and make it difficult to attribute
changes to individual parameters.

A one-factor-at-a-time design was therefore used around the validated baseline:

```text
neighbor_size: 100
sample_size: 500
sampling: recent
similarity: vec
session_weighting: div
score_weighting: div
```

The compact grid contained the baseline plus eight variants:

- `neighbor_size: 200`;
- `sample_size: 250`;
- `sample_size: 1000`;
- `similarity: cosine`;
- `session_weighting: same`;
- `session_weighting: quadratic`;
- `score_weighting: same`;
- `score_weighting: quadratic`.

### Resumable Runner

Added:

```text
src/recbole_framework/tuning/tune_vsknn_audited.py
```

Run command:

```powershell
python -m src.recbole_framework.tuning.tune_vsknn_audited
```

The runner:

- reuses the three already validated baseline results;
- creates stable SHA-256-based run IDs;
- writes the CSV after every configuration;
- skips successful run IDs after restart;
- records failures without losing other progress;
- automatically exports the best MRR@10 row per dataset.

The three baseline rows plus 24 new configurations produced 27 successful
result rows. No run failed. A second invocation reported zero pending runs,
confirming resume behavior.

### Best MRR@10 Configuration per Dataset

| Dataset | Neighbor size | Sample size | Similarity | Session weighting | Score weighting | Hit@10 | NDCG@10 | MRR@10 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| Yoochoose sample | 100 | 500 | vec | div | quadratic | 0.5377 | 0.3478 | 0.2880 |
| Globo sample | 100 | 1000 | vec | div | div | 0.3298 | 0.1341 | 0.0751 |
| Adressa sample | 200 | 500 | vec | div | div | 0.4313 | 0.2312 | 0.1703 |

### Change Relative to the Audited Baseline

| Dataset | Baseline MRR@10 | Best MRR@10 | Absolute change |
| --- | ---: | ---: | ---: |
| Yoochoose sample | 0.2867 | 0.2880 | +0.0013 |
| Globo sample | 0.0697 | 0.0751 | +0.0054 |
| Adressa sample | 0.1650 | 0.1703 | +0.0053 |

Yoochoose's MRR-optimal quadratic score decay slightly reduces Hit@10 from
0.5387 to 0.5377 while improving NDCG@10 from 0.3470 to 0.3478. The result is
therefore explicitly described as MRR-optimal rather than universally best.

Globo benefits most from a larger recent candidate sample. Reducing the sample
to 250 substantially lowers MRR@10 to 0.0592, while increasing it to 1000 raises
MRR@10 to 0.0751.

Adressa benefits from retaining 200 final neighbors, reaching MRR@10 0.1703.
Using only 250 candidate sessions offers an efficiency-oriented alternative
with MRR@10 0.1682 and a runtime of 82.03 seconds.

### Cross-Dataset Interpretation

- `session_weighting: same` reduced MRR on all three datasets, supporting
  position-aware current-session weighting as a default.
- cosine was not the best similarity on any dataset; `vec` remains the most
  defensible reference default.
- optimal neighborhood/sample capacity is domain-dependent.
- larger candidate sets do not universally improve results: they help Globo,
  provide only a small Yoochoose gain, and are weaker than more final neighbors
  on Adressa.
- parameter selection must state the optimization metric because MRR, Hit, and
  runtime can prefer different configurations.

### Output Files

```text
recbole_results/vsknn_audited/compact_tuning_results.csv
recbole_results/vsknn_audited/compact_tuning_best_by_dataset.csv
```

### Validation

- 23 automated algorithm, candidate-order, CLI, grid, run-ID, resume, and
  best-selection tests passed before tuning;
- all 24 new RecBole runs completed successfully;
- baseline rows were not recomputed;
- result selection was based on MRR@10, matching the thesis primary metric;
- resume behavior was verified after completion.

### Next Steps

1. Use these compact-grid winners as the audited VSKNN sample results.
2. Open the prepared official RecBole proposal issue.
3. Ask maintainers whether the standard Trainer plus zero-loss parameter is
   acceptable for a non-parametric model.
4. Only if required for the thesis, run the selected configurations on full
   datasets using consistent server hardware.

## 2026-07-17: Consolidated Best-Model Overview

### Why This Was Added

Results were distributed across the Top-N tuning CSV, the session-model tuning
CSV, and the new audited VSKNN tuning output. A single reproducible overview was
needed so that every model/dataset combination, the selection rule, and each
dataset winner can be cited without manually copying values.

### Code and Documentation Changes

Added:

```text
src/recbole_framework/analysis/build_best_model_overview.py
tests/test_best_model_overview.py
docs/audited_vsknn_and_best_models_overview.md
recbole_results/summary/best_models_by_dataset.csv
recbole_results/summary/best_overall_model_per_dataset.csv
```

The analysis script cleans failed or malformed rows, converts metric columns to
numeric values, and selects the best successful configuration for each
model/dataset pair by MRR@10. Equal MRR@10 values are resolved by lower recorded
runtime and then stable source order. Legacy `VS-KNN` rows are excluded and
replaced with the compact-tuned audited `VSKNN` results so that old and corrected
implementations are not mixed.

The generated Markdown document also records what changed in VSKNN and why:
reference position weighting and neighbor decay were restored, duplicated
prefix sessions were collapsed, the index was kept training-only, the
thesis-specific popularity correction was removed from the upstream-faithful
path, names were standardized, and candidate retrieval was optimized without
changing recommendation quality.

### Audited VSKNN Versus Legacy Run

| Dataset | Metric | Legacy | Audited tuned | Absolute change | Relative MRR change |
| --- | --- | ---: | ---: | ---: | ---: |
| Yoochoose sample | Hit@10 | 0.4947 | 0.5377 | +0.0430 | — |
| Yoochoose sample | NDCG@10 | 0.3177 | 0.3478 | +0.0301 | — |
| Yoochoose sample | MRR@10 | 0.2624 | 0.2880 | +0.0256 | +9.8% |
| Globo sample | Hit@10 | 0.3373 | 0.3298 | -0.0075 | — |
| Globo sample | NDCG@10 | 0.1326 | 0.1341 | +0.0015 | — |
| Globo sample | MRR@10 | 0.0713 | 0.0751 | +0.0038 | +5.3% |
| Adressa sample | Hit@10 | 0.3428 | 0.4313 | +0.0885 | — |
| Adressa sample | NDCG@10 | 0.1735 | 0.2312 | +0.0577 | — |
| Adressa sample | MRR@10 | 0.1225 | 0.1703 | +0.0478 | +39.0% |

These differences combine the correctness changes and the subsequent compact
tuning; they cannot be interpreted as the isolated effect of one code change.
Globo is the only dataset where tuned audited VSKNN has lower Hit@10, while its
ranking quality measured by NDCG@10 and the primary metric MRR@10 improves.

### Overall Dataset Winners by MRR@10

| Scenario | Dataset | Best model | MRR@10 |
| --- | --- | --- | ---: |
| Session-based | Adressa sample | GRU4Rec | 0.2181 |
| Session-based | Globo sample | GRU4Rec | 0.1493 |
| Session-based | Yoochoose sample | VSKNN | 0.2880 |
| Top-N | Amazon | BPR | 0.0519 |
| Top-N | MovieLens | BPR | 0.1546 |

The detailed CSV contains 26 best model/dataset rows: six session models on
each of three sample datasets and four Top-N models on each of two datasets.
Top-N and session-based scenarios are reported separately and are not ranked
against each other. Runtime remains descriptive because audited VSKNN ran on
CPU while many stored framework runs used CUDA.

### Reproduction and Validation

Regenerate all three overview artifacts with:

```powershell
python -m src.recbole_framework.analysis.build_best_model_overview
```

The generator produced exactly 26 model/dataset rows and five dataset winners.
Two additional automated tests verified malformed-row filtering and the
MRR@10/runtime tie-breaking rule.
