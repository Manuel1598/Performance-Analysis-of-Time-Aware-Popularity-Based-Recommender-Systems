# Feature proposal: add Vector Multiplication Session-based kNN (VSKNN)

## Description

I would like to contribute VSKNN as a non-parametric sequential recommender.
VSKNN is a strong and computationally simple session-based baseline that gives
more importance to recent interactions in the current session when selecting
and scoring neighboring sessions.

## Reference

Malte Ludewig and Dietmar Jannach. 2018. *Evaluation of Session-based
Recommendation Algorithms*. User Modeling and User-Adapted Interaction.
https://doi.org/10.1007/s11257-018-9209-6

Reference implementation:
https://github.com/rn5l/session-rec/blob/master/algorithms/knn/vsknn.py

## Proposed RecBole integration

- inherit from `SequentialRecommender`;
- build the neighbor index strictly from `train_data.dataset`;
- reconstruct one training session from RecBole's augmented prefix-target rows;
- implement `predict` and `full_sort_predict`;
- support recent candidate sampling, vector-multiplication/weighted-cosine
  similarity, positional session weighting, and neighbor score weighting;
- add a default model YAML, documentation, and deterministic unit tests.

Suggested defaults:

```yaml
neighbor_size: 100
sample_size: 1000
sampling: recent
similarity: vec
session_weighting: div
score_weighting: div
```

## Evaluation and correctness checks

The contribution will include tests for hand-calculated similarities and item
scores, reconstruction of augmented training sessions, deterministic recent
sampling, empty sessions, and validation/test leakage. I plan to report Hit@10,
MRR@10, NDCG@10, runtime, and peak memory on at least two public session-based
datasets and compare against Pop, GRU4Rec, and another sequential baseline.

An initial RecBole 1.2.1 CPU smoke test on `yoochoose_recbole_sample` with seed
42, `neighbor_size: 100`, and `sample_size: 500` completed successfully. It
produced Hit@10 `0.5387`, NDCG@10 `0.3470`, and MRR@10 `0.2867`. The previous
SKNN-like implementation produced `0.4947`, `0.3177`, and `0.2624` under the
same data split and ranking parameters. These preliminary figures are included
as a correctness signal; a multi-dataset benchmark will follow in the PR.

The identical smoke-test configuration also completed on Globo and Adressa:

| Dataset | Hit@10 | NDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| Yoochoose sample | 0.5387 | 0.3470 | 0.2867 |
| Globo sample | 0.2916 | 0.1211 | 0.0697 |
| Adressa sample | 0.4276 | 0.2259 | 0.1650 |

The mixed cross-domain change relative to the former SKNN-like implementation
motivates reporting the full benchmark rather than selecting only Yoochoose.

A compact one-factor-at-a-time tuning study (nine configurations per dataset,
including the validated baseline) subsequently completed without failures. The
best MRR@10 configurations were:

| Dataset | Key change from default | Hit@10 | NDCG@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: |
| Yoochoose sample | `score_weighting: quadratic` | 0.5377 | 0.3478 | 0.2880 |
| Globo sample | `sample_size: 1000` | 0.3298 | 0.1341 | 0.0751 |
| Adressa sample | `neighbor_size: 200` | 0.4313 | 0.2312 | 0.1703 |

The complete grid and stable run IDs are retained as reproducibility artifacts.

## Design question for maintainers

VSKNN has no gradient-based training phase. Is the preferred upstream design to
keep a zero-loss compatibility parameter with the standard `Trainer`, or would
the maintainers prefer a dedicated non-parametric trainer/lifecycle hook?

I am happy to adapt the implementation and configuration names before opening
the pull request.
