# VSKNN audit for RecBole upstream preparation

## Reference basis

- Ludewig and Jannach, *Evaluation of Session-based Recommendation Algorithms*
  (User Modeling and User-Adapted Interaction, 2018; arXiv:1803.09587).
- The authors' `rn5l/session-rec` `VMContextKNN` implementation, treated as the
  executable reference for weighting and candidate scoring.
- RecBole's `SequentialDataset.data_augmentation`, which converts each original
  session into multiple prefix-target rows before splitting.

## Findings in the previous implementation

1. It used unweighted binary cosine similarity. This is SKNN-like behavior, not
   the defining vector-multiplication variant of VSKNN.
2. It did not weight current-session clicks by position.
3. It did not down-weight a neighbor according to the recency of its last item
   shared with the current session.
4. Every augmented prefix-target row became an independent reference session.
   Longer sessions were therefore duplicated and systematically over-weighted.
5. The model was correctly constructed from `train_data.dataset`, so validation
   and test rows were not directly indexed. The new implementation makes this
   boundary explicit by collapsing only rows received from that training split.
6. `vsknn_popularity_weight` is a thesis-specific extension and is deliberately
   excluded from the upstream-faithful VSKNN implementation.

## Implemented behavior

- one reconstructed reference session per RecBole session/user ID;
- recent candidate-session sampling;
- `vec` (default) and weighted `cosine` similarities;
- `same`, `linear`, `div`, `log`, and `quadratic` positional weightings;
- deterministic neighbor tie-breaking;
- compatibility aliases for the thesis' old class/configuration keys.

## Split and leakage conclusion

With `eval_args.order: TO`, RecBole first creates ordered prefix-target examples
and then splits those examples. Constructing VSKNN with `train_data.dataset`
therefore exposes only training rows. Collapsing those rows selects the longest
available training prefix plus its training target for each session. Validation
and test targets are never read by the neighbor index.

This protects the evaluation boundary but means RecBole's ratio split truncates
each indexed training session at its split boundary. This is expected: later
events belong to validation/test and must not be used as neighbors.

## Preferred configuration

```yaml
neighbor_size: 100
sample_size: 1000
sampling: recent
similarity: vec
session_weighting: div
score_weighting: div
```

Legacy `vsknn_k` and `vsknn_sample_size` remain accepted temporarily. New code
should use the preferred names above.

## RecBole 1.2.1 smoke-test result

The audited implementation was run on `yoochoose_recbole_sample` on CPU with
seed 42, `neighbor_size: 100`, and `sample_size: 500`. The legacy comparison is
the existing result produced with the same dataset, seed, neighbor size, sample
size, split configuration, and ranking metrics.

| Metric | Legacy implementation | Audited VSKNN | Absolute change |
| --- | ---: | ---: | ---: |
| Hit@10 | 0.4947 | 0.5387 | +0.0440 |
| NDCG@10 | 0.3177 | 0.3470 | +0.0293 |
| MRR@10 | 0.2624 | 0.2867 | +0.0243 |

These values are a correctness smoke test, not yet a publication benchmark.
Runtime hardware differs from the earlier CUDA experiment, so runtime values
must not be compared until both variants are rerun on identical hardware.

### Cross-domain sample results

All three local session samples were then evaluated in one RecBole 1.2.1 CPU
run with the same seed and configuration.

| Dataset | Hit@10 | NDCG@10 | MRR@10 | Reference sessions | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yoochoose sample | 0.5387 | 0.3470 | 0.2867 | 126,496 | 134.90 |
| Globo sample | 0.2916 | 0.1211 | 0.0697 | 175,715 | 121.94 |
| Adressa sample | 0.4276 | 0.2259 | 0.1650 | 98,032 | 147.53 |

Compared with the stored legacy `neighbor_size=100`, `sample_size=500` runs,
the audited implementation improves all three primary metrics on Yoochoose and
Adressa. Globo decreases from Hit@10 0.3373, NDCG@10 0.1326, and MRR@10 0.0713.
This mixed result shows that the correction is not merely a universal score
increase and that multi-domain reporting is necessary.

### Performance optimization

Profiling 2,000 Adressa test queries showed that the original audited adapter
spent 36.73 of 43.32 seconds selecting recent candidates. It constructed a
large union and globally sorted it for every query, producing about 59 million
sort-key calls.

The optimized adapter pre-sorts the session list for each item once and merges
only the relevant lists until `sample_size` unique sessions have been found. It
also prepares current-session position weights and reverse item positions once
per query instead of once per candidate/neighbor.

The same 2,000-query profile decreased from 43.321 to 8.076 seconds (5.36x),
while 19 unit and regression tests confirmed identical candidate ordering and
scores. Full sample metrics remained exactly unchanged. End-to-end runtimes
changed as follows:

| Dataset | Before (s) | After (s) | Reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Yoochoose sample | 169.39 | 134.90 | 20.4% | 1.26x |
| Globo sample | 218.84 | 121.94 | 44.3% | 1.79x |
| Adressa sample | 2,002.70 | 147.53 | 92.6% | 13.57x |

The large variation confirms that candidate overlap, rather than reference
session count alone, determines how much the recency-merge optimization helps.
