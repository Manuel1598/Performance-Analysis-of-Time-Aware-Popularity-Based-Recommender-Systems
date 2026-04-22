# RecBole Model Integration

This document describes how custom recommendation models are implemented within the RecBole framework in this project.

---

## Motivation

To ensure standardized evaluation, reproducibility, and comparability, all models are integrated directly into the RecBole framework.

Instead of maintaining a standalone pipeline, RecBole is used as the central system for:

* data handling
* model execution
* evaluation

Custom models (**MostPop**, **RecentPop**, **DecayPop**) are implemented as native RecBole models.

---

## RecBole Model Structure

According to the official RecBole developer documentation, custom models follow a standardized structure.

A new model is implemented as a class that inherits from one of the RecBole base classes:

* `GeneralRecommender`
* `ContextRecommender`
* `SequentialRecommender`
* `KnowledgeRecommender`

For this project, **GeneralRecommender** is used, as the focus is on Top-N recommendation.

---

## Required Components of a Custom Model

Each RecBole model must define the following elements:

### 1. Model Class Definition

```python
class MyModel(GeneralRecommender):
    ...
```

---

### 2. Input Type

The model must define its training input type:

* `POINTWISE`
* `PAIRWISE`

Example:

```python
input_type = InputType.POINTWISE
```

---

### 3. `__init__()` Method

Responsible for:

* loading configuration parameters
* accessing dataset information
* initializing model-specific structures

---

### 4. `calculate_loss()`

Defines the training objective.

* required for RecBole training loop
* may be simplified or adapted for non-learning models

---

### 5. `predict()`

Returns prediction scores for given user-item pairs.

---

### 6. `full_sort_predict()`

This method is essential for Top-N recommendation.

It computes scores for **all items for a given user**, enabling ranking and evaluation.

---

## RecBole Training and Evaluation Pipeline

After implementing the model, RecBole provides a standard pipeline:

```python
config = Config(model='MyModel', dataset='dataset_name')
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = MyModel(config, train_data.dataset)
trainer = Trainer(config, model)

trainer.fit(train_data, valid_data)
trainer.evaluate(test_data)
```

---

## Application in This Project

In this thesis, the following models are implemented using this structure:

* MostPop
* RecentPop
* DecayPop

### Key Characteristics

* implemented as subclasses of `GeneralRecommender`
* integrated into RecBole’s training and evaluation pipeline
* compatible with RecBole datasets and configuration system

---

## Special Considerations for Popularity-Based Models

Unlike traditional machine learning models, popularity-based models:

* do not require iterative training
* rely on statistics computed from the dataset

### Therefore:

* `calculate_loss()` is implemented as a minimal or dummy function
* core logic is implemented in `full_sort_predict()`
* popularity scores are computed during initialization

---

## Summary

Custom model integration in RecBole follows a clear and structured process:

* define a model class based on a RecBole base class
* implement required methods (`__init__`, `calculate_loss`, `predict`, `full_sort_predict`)
* integrate into the RecBole pipeline using `Config` and `Trainer`

### This approach ensures:

* reproducibility
* comparability across models
* extensibility for future work





## RecBole Execution Pipeline

To run experiments within RecBole, a standardized execution pipeline is used.

A typical workflow consists of the following steps:

```python
config = Config(...)
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = MostPopRecBole(config, dataset)
trainer = Trainer(config, model)

trainer.fit(train_data, valid_data)
trainer.evaluate(test_data)
```

These components form the core building blocks of the RecBole framework.

---

## Key RecBole Components

### 1. Config

The configuration object defines all relevant settings for an experiment:

* model selection
* dataset
* field mappings (user, item, timestamp)
* evaluation settings
* top-k values
* metrics

This allows flexible and reproducible experimentation.

---

### 2. create_dataset(config)

This function loads the dataset in RecBole format.

**Important:**

* the dataset must be provided as a `.inter` file
* raw CSV files are not directly used in the RecBole pipeline

---

### 3. data_preparation(config, dataset)

This step handles:

* dataset splitting
* creation of training, validation, and test loaders
* internal data structures

**Important:**

For this project, it is necessary to ensure that the splitting strategy
(e.g., chronological leave-one-out) is aligned with the thesis methodology.

---

### 4. Trainer

The Trainer component manages:

* model training
* evaluation
* logging

Even for non-learning models such as **MostPop**, this step is required for compatibility with the RecBole framework.

---

## Implications for Model Design

Custom models in this project are designed to follow a strict separation of responsibilities.

### Model Responsibilities

The model (e.g., `MostPopRecBole`) is responsible for:

* receiving config and dataset
* computing internal statistics (e.g., item popularity)
* generating prediction scores

**It does not:**

* load raw files
* handle preprocessing
* perform data splitting

---

### Responsibilities Outside the Model

The surrounding pipeline is responsible for:

* preparing `.inter` files
* defining configuration settings
* running experiments via RecBole

---

### Responsibilities of RecBole

RecBole handles:

* dataset loading
* data splitting and data loaders
* training and evaluation
* metric computation

---

## Summary

This separation ensures:

* clean architecture
* reproducibility
* compatibility with RecBole
* easy extension to additional models and datasets

