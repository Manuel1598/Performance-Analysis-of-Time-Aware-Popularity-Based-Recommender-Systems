import math
import unittest

from src.recbole_framework.custom_models.session.vsknn_core import (
    collapse_augmented_sessions,
    position_weights,
    recent_item_steps,
    score_decay,
    score_decay_from_steps,
    score_neighbors,
    score_neighbors_from_steps,
    session_similarity,
    weighted_session_similarity,
)


class VSKNNCoreTests(unittest.TestCase):
    def test_div_weights_favor_recent_clicks(self):
        self.assertEqual(position_weights([10, 20, 30], "div"), {10: 1 / 3, 20: 2 / 3, 30: 1.0})

    def test_vec_similarity_matches_reference_formula(self):
        similarity = session_similarity([10, 20, 30], {20, 30, 40}, "div", "vec")
        self.assertAlmostEqual(similarity, (2 / 3 + 1.0) / 3)

    def test_precomputed_similarity_is_identical(self):
        current = [10, 20, 30]
        neighbor = {20, 30, 40}
        expected = session_similarity(current, neighbor, "div", "vec")
        actual = weighted_session_similarity(
            position_weights(current, "div"), neighbor, "vec"
        )
        self.assertEqual(actual, expected)

    def test_cosine_similarity_matches_weighted_formula(self):
        similarity = session_similarity([10, 20], {20, 30}, "div", "cosine")
        expected = 1.0 / (math.sqrt(0.5**2 + 1.0**2) * math.sqrt(2))
        self.assertAlmostEqual(similarity, expected)

    def test_cosine_similarity_with_zero_weight_norm_returns_zero(self):
        similarity = session_similarity([10] * 11, {10, 20}, "linear", "cosine")
        self.assertEqual(similarity, 0.0)

    def test_score_decay_uses_most_recent_shared_click(self):
        self.assertEqual(score_decay([10, 20, 30], {10, 40}, "div"), 1 / 3)
        self.assertEqual(score_decay([10, 20, 30], {20, 40}, "div"), 1 / 2)
        self.assertEqual(score_decay([10, 20, 30], {30, 40}, "div"), 1.0)

    def test_neighbor_item_scores_match_hand_calculation(self):
        scores = score_neighbors(
            [10, 20, 30],
            [({30, 40}, 0.6), ({10, 40, 50}, 0.9)],
            "div",
        )
        self.assertAlmostEqual(scores[30], 0.6)
        self.assertAlmostEqual(scores[10], 0.3)
        self.assertAlmostEqual(scores[40], 0.9)
        self.assertAlmostEqual(scores[50], 0.3)

    def test_precomputed_score_decay_and_scores_are_identical(self):
        current = [10, 20, 10, 30]
        neighbors = [({30, 40}, 0.6), ({10, 40, 50}, 0.9)]
        steps = recent_item_steps(current)
        self.assertEqual(
            score_decay_from_steps(steps, {10, 40}, "div"),
            score_decay(current, {10, 40}, "div"),
        )
        self.assertEqual(
            score_neighbors_from_steps(steps, neighbors, "div"),
            score_neighbors(current, neighbors, "div"),
        )

    def test_augmented_rows_are_collapsed_per_session(self):
        rows = [
            (1, [10, 0, 0], 1, 20, 2.0),
            (1, [10, 20, 0], 2, 30, 3.0),
            (2, [40, 0, 0], 1, 50, 5.0),
        ]
        self.assertEqual(
            collapse_augmented_sessions(rows),
            [(1, [10, 20, 30], 3.0), (2, [40, 50], 5.0)],
        )

    def test_only_passed_training_rows_can_enter_reference_sessions(self):
        training_rows = [(1, [10, 0], 1, 20, 2.0)]
        validation_target = 30
        sessions = collapse_augmented_sessions(training_rows)
        self.assertNotIn(validation_target, sessions[0][1])

    def test_duplicate_click_uses_last_position(self):
        self.assertEqual(position_weights([10, 20, 10], "div")[10], 1.0)

    def test_invalid_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            position_weights([10], "unknown")


if __name__ == "__main__":
    unittest.main()
