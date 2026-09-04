import unittest
from unittest.mock import patch
from collections import defaultdict
from types import SimpleNamespace

import torch
from recbole.model.abstract_recommender import SequentialRecommender

from src.recbole_framework.custom_models.session.vstan_recbole import VSTANRecBole


class VSTANReferenceSessionTests(unittest.TestCase):
    @staticmethod
    def _model() -> VSTANRecBole:
        model = VSTANRecBole.__new__(VSTANRecBole)
        model.USER_ID = "session_id"
        model.ITEM_SEQ = "item_seq"
        model.ITEM_SEQ_LEN = "item_seq_len"
        model.ITEM_ID = "item_id"
        model.time_field = "timestamp"
        model.n_items = 60
        model.reference_sessions = []
        model.reference_session_sets = []
        model.reference_session_timestamps = []
        model.item_sessions = defaultdict(set)
        return model

    @staticmethod
    def _training_dataset():
        return SimpleNamespace(
            inter_feat={
                "session_id": torch.tensor([1, 1, 2]),
                "item_seq": torch.tensor(
                    [
                        [10, 0, 0],
                        [10, 20, 0],
                        [40, 0, 0],
                    ]
                ),
                "item_seq_len": torch.tensor([1, 2, 1]),
                "item_id": torch.tensor([20, 30, 50]),
                "timestamp": torch.tensor([2.0, 3.0, 5.0]),
            }
        )

    def test_constructor_builds_sessions_before_popularity_statistics(self):
        def initialise_recbole_base(model, config, dataset):
            torch.nn.Module.__init__(model)
            model.USER_ID = "session_id"
            model.ITEM_SEQ = "item_seq"
            model.ITEM_SEQ_LEN = "item_seq_len"
            model.ITEM_ID = "item_id"
            model.n_items = 60

        config = {
            "device": "cpu",
            "TIME_FIELD": "timestamp",
            "vstan_k": 100,
            "vstan_sample_size": 1000,
            "vstan_position_decay": 0.1,
            "vstan_idf_weighting": True,
            "vstan_popularity_weight": 0.0,
        }
        with patch.object(
            SequentialRecommender,
            "__init__",
            new=initialise_recbole_base,
        ):
            model = VSTANRecBole(config, self._training_dataset())

        self.assertEqual(model.reference_sessions, [[10, 20, 30], [40, 50]])
        self.assertAlmostEqual(float(model.item_popularity[10]), 1.0 / 5.0)
    def test_builds_one_reference_per_original_training_session(self):
        model = self._model()

        model._build_reference_sessions(self._training_dataset())

        self.assertEqual(model.reference_sessions, [[10, 20, 30], [40, 50]])
        self.assertEqual(model.reference_session_timestamps, [3.0, 5.0])
        self.assertEqual(model.item_sessions[10], {0})
        self.assertEqual(model.item_sessions[20], {0})
        self.assertEqual(model.item_sessions[30], {0})
        self.assertEqual(model.item_sessions[40], {1})
        self.assertEqual(model.item_sessions[50], {1})

    def test_popularity_uses_collapsed_reference_events_once(self):
        model = self._model()
        model._build_reference_sessions(self._training_dataset())

        popularity = model._compute_item_popularity()

        for item_id in [10, 20, 30, 40, 50]:
            self.assertAlmostEqual(float(popularity[item_id]), 1.0 / 5.0)

    def test_rows_not_passed_from_training_cannot_enter_reference_index(self):
        model = self._model()
        validation_target = 55

        model._build_reference_sessions(self._training_dataset())

        self.assertNotIn(validation_target, model.reference_sessions[0])
        self.assertNotIn(validation_target, model.item_sessions)


if __name__ == "__main__":
    unittest.main()
