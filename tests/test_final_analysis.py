import unittest

import numpy as np
import pandas as pd
import torch
from recbole.data.interaction import Interaction

from tools.consolidate_final_results import source_for, select_one, validate_final
from tools.evaluate_session_prefix_groups import prefix_group, metrics_from_ranks, reconcile, METRICS, score_batch


class FixedScores(torch.nn.Module):
    ITEM_SEQ_LEN = "length"
    ITEM_SEQ = "items"

    def full_sort_predict(self, interaction):
        scores = torch.arange(12, dtype=torch.float32).repeat(len(interaction), 1)
        scores[:, 0] = 1000
        return scores


class FinalAnalysisTests(unittest.TestCase):
    def test_scoring_masks_padding_and_keeps_query_order(self):
        interaction = Interaction({"length": torch.tensor([2,3]), "items": torch.tensor([[1,2,0],[1,2,3]])})
        batch = (interaction, (torch.tensor([0]), torch.tensor([11])), torch.tensor([0,1]), torch.tensor([10,8]))
        lengths, _, targets, ranks = score_batch(batch, FixedScores(), 12)
        np.testing.assert_array_equal(lengths, [2,3])
        np.testing.assert_array_equal(targets, [10,8])
        np.testing.assert_array_equal(ranks, [1,4])
        invalid = (interaction, None, torch.tensor([0,0]), torch.tensor([10,8]))
        with self.assertRaises(ValueError):
            score_batch(invalid, FixedScores(), 12)

    def test_version_policy(self):
        self.assertEqual(source_for("VSTAN"), ("pc1_vstan", "validation_first_v7_vstan_collapsed"))
        self.assertEqual(source_for("VS-KNN"), ("pc2", "validation_first_v6"))

    def test_ties_and_failed_candidates(self):
        frame = pd.DataFrame([
            dict(run_id="b", status="success", **{"valid_mrr@10": .5}, runtime_seconds=3),
            dict(run_id="a", status="success", **{"valid_mrr@10": .5}, runtime_seconds=3),
        ])
        self.assertEqual(select_one(frame).run_id, "a")
        frame.loc[0, "status"] = "failed"
        with self.assertRaises(ValueError):
            select_one(frame)

    def test_groups(self):
        self.assertEqual([prefix_group(n) for n in (1, 2, 3, 4, 20)], ["1", "2", "3", "4+", "4+"])
        with self.assertRaises(ValueError):
            prefix_group(0)

    def test_single_target_metrics_and_cutoffs(self):
        ranks = np.array([1, 2, 10, 0])
        at10 = metrics_from_ranks(ranks, 10)
        self.assertEqual(at10["hit@10"], .75)
        self.assertAlmostEqual(at10["mrr@10"], .4)
        self.assertAlmostEqual(at10["ndcg@10"], (1 + 1/np.log2(3) + 1/np.log2(11))/4)
        self.assertEqual(metrics_from_ranks(ranks, 5)["hit@5"], .5)
        self.assertTrue(np.isnan(metrics_from_ranks(np.array([]), 10)["hit@10"]))

    def test_reconciliation_rejects_changed_scores_and_counts(self):
        values = {m: .12344 for m in METRICS}
        original = {m: .1234 for m in METRICS}
        original["recommendation_count"] = 100
        reconcile(values, original, 10)
        with self.assertRaises(ValueError):
            reconcile(values, original, 9)
        values["mrr@10"] = .13
        with self.assertRaises(ValueError):
            reconcile(values, original, 10)


if __name__ == "__main__":
    unittest.main()
