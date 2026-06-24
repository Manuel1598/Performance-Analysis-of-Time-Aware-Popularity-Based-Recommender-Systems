# Presentation Summary - RecBole Results

## Best Model Per Dataset
| experiment_type | dataset | model | mrr@10 | hit@10 | ndcg@10 | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| session | adressa_recbole_sample | GRU4Rec | 0.2181 | 0.522 | 0.2892 | 26.18 |
| session | globo_recbole_sample | GRU4Rec | 0.1493 | 0.3614 | 0.1988 | 27.36 |
| session | yoochoose_recbole_sample | VSTAN | 0.2773 | 0.5334 | 0.3384 | 541.25 |
| topn | amazon_recbole | BPR | 0.0519 | 0.0754 | 0.0573 | 12903.1 |
| topn | movielens_recbole | BPR | 0.1546 | 0.3546 | 0.084 | 5134.07 |

## Session Ranking
| dataset | model | mrr@10 | hit@10 | ndcg@10 | runtime_seconds | rank_by_mrr@10 |
| --- | --- | --- | --- | --- | --- | --- |
| adressa_recbole_sample | GRU4Rec | 0.2181 | 0.522 | 0.2892 | 26.18 | 1 |
| adressa_recbole_sample | VSTAN | 0.1604 | 0.4308 | 0.2229 | 698.26 | 24 |
| adressa_recbole_sample | VS-KNN | 0.1225 | 0.3428 | 0.1735 | 659.7 | 91 |
| globo_recbole_sample | GRU4Rec | 0.1493 | 0.3614 | 0.1988 | 27.36 | 1 |
| globo_recbole_sample | VSTAN | 0.0831 | 0.3644 | 0.1482 | 287.73 | 19 |
| globo_recbole_sample | VS-KNN | 0.0787 | 0.3597 | 0.1436 | 738.02 | 33 |
| yoochoose_recbole_sample | VSTAN | 0.2773 | 0.5334 | 0.3384 | 541.25 | 1 |
| yoochoose_recbole_sample | VS-KNN | 0.2625 | 0.4987 | 0.3187 | 458.5 | 52 |
| yoochoose_recbole_sample | GRU4Rec | 0.2469 | 0.4818 | 0.3027 | 32.13 | 64 |

## Interpretation Notes
- `MRR@10` is the primary ranking metric.
- Session popularity baselines are included as MostPop, RecentPop, and DecayPop.
- VS-KNN and VSTAN are compared against GRU4Rec under the same session protocol.
- Use runtime tables to discuss quality-runtime tradeoffs.