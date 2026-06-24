# Popularity Weighting Evaluation

This table compares the best unweighted and weighted VS-KNN/VSTAN configuration per dataset.

| dataset | model | best_unweighted_mrr@10 | best_weighted_mrr@10 | delta_mrr@10_weighted_minus_unweighted | weighted_improves_mrr@10 | best_weighted_popularity_weight |
| --- | --- | --- | --- | --- | --- | --- |
| adressa_recbole_sample | VS-KNN | 0.1497 | 0.0858 | -0.0639 | False | 0.5 |
| adressa_recbole_sample | VSTAN | 0.1619 | 0.0966 | -0.06529999999999998 | False | 0.5 |
| globo_recbole_sample | VS-KNN | 0.0555 | 0.0241 | -0.0314 | False | 0.5 |
| globo_recbole_sample | VSTAN | 0.0563 | 0.0278 | -0.028500000000000004 | False | 0.5 |
| yoochoose_recbole_sample | VS-KNN | 0.2554 | 0.2477 | -0.007700000000000012 | False | 0.5 |
| yoochoose_recbole_sample | VSTAN | 0.2676 | 0.2601 | -0.007500000000000007 | False | 0.5 |

Interpretation: negative delta values mean that popularity weighting reduced ranking quality compared with the unweighted variant.