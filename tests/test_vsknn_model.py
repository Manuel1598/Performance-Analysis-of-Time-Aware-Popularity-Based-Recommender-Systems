import unittest

from src.recbole_framework.custom_models.session.vsknn_recbole import VSKNN


class VSKNNCandidateSamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(VSKNN)
        self.model.sample_size = 3
        self.model.reference_session_timestamps = [10.0, 30.0, 20.0, 40.0]
        self.model.item_sessions_by_recency = {
            10: [3, 1, 0],
            20: [3, 2, 0],
        }

    def test_recency_merge_matches_global_union_sort(self):
        expected = [3, 1, 2]
        self.assertEqual(self.model._find_candidate_sessions({10, 20}), expected)

    def test_duplicate_sessions_are_returned_once(self):
        candidates = self.model._find_candidate_sessions({10, 20})
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_zero_sample_size_returns_all_candidates_in_recency_order(self):
        self.model.sample_size = 0
        self.assertEqual(
            self.model._find_candidate_sessions({10, 20}),
            [3, 1, 2, 0],
        )

    def test_unknown_items_return_no_candidates(self):
        self.assertEqual(self.model._find_candidate_sessions({999}), [])


if __name__ == "__main__":
    unittest.main()
