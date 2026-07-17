"""RecBole adapter for Vector Multiplication Session-based kNN (VSKNN)."""

from collections import defaultdict
from typing import Optional

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

from .vsknn_core import (
    SIMILARITIES,
    WEIGHTING_FUNCTIONS,
    collapse_augmented_sessions,
    score_neighbors,
    session_similarity,
)


class VSKNN(SequentialRecommender):
    """Non-parametric VSKNN using training sessions as the neighbor index.

    Preferred configuration keys follow the reference implementation. Legacy
    ``vsknn_*`` keys remain accepted so existing thesis experiments reproduce.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.device = config["device"]
        self.time_field = config["TIME_FIELD"]
        self.neighbor_size = int(
            self._config_value(config, "neighbor_size", "vsknn_k", 100)
        )
        self.sample_size = int(
            self._config_value(config, "sample_size", "vsknn_sample_size", 1000)
        )
        self.sampling = str(self._config_value(config, "sampling", None, "recent"))
        self.similarity = str(self._config_value(config, "similarity", None, "vec"))
        self.session_weighting = str(
            self._config_value(config, "session_weighting", None, "div")
        )
        self.score_weighting = str(
            self._config_value(config, "score_weighting", None, "div")
        )
        self._validate_config()

        # Trainer expects a differentiable loss even though VSKNN has no training.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.reference_sessions: list[list[int]] = []
        self.reference_session_sets: list[set[int]] = []
        self.reference_session_timestamps: list[float] = []
        self.item_sessions: dict[int, set[int]] = defaultdict(set)
        self._build_reference_sessions(dataset)

    @staticmethod
    def _config_value(config, preferred: str, legacy: Optional[str], default):
        try:
            value = config[preferred]
        except KeyError:
            value = None
        if value is not None:
            return value
        if legacy is not None:
            try:
                value = config[legacy]
            except KeyError:
                value = None
            if value is not None:
                return value
        return default

    def _validate_config(self) -> None:
        if self.neighbor_size <= 0:
            raise ValueError("neighbor_size must be greater than zero")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if self.sampling != "recent":
            raise ValueError("Only deterministic recent sampling is supported")
        if self.similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {sorted(SIMILARITIES)}")
        if self.session_weighting not in WEIGHTING_FUNCTIONS:
            raise ValueError(
                f"session_weighting must be one of {sorted(WEIGHTING_FUNCTIONS)}"
            )
        if self.score_weighting not in WEIGHTING_FUNCTIONS:
            raise ValueError(
                f"score_weighting must be one of {sorted(WEIGHTING_FUNCTIONS)}"
            )

    def _build_reference_sessions(self, dataset) -> None:
        inter_feat = dataset.inter_feat
        rows = zip(
            inter_feat[self.USER_ID].cpu().tolist(),
            inter_feat[self.ITEM_SEQ].cpu().tolist(),
            inter_feat[self.ITEM_SEQ_LEN].cpu().tolist(),
            inter_feat[self.ITEM_ID].cpu().tolist(),
            inter_feat[self.time_field].cpu().tolist(),
        )
        sessions = collapse_augmented_sessions(rows)

        for _, items, timestamp in sessions:
            session_index = len(self.reference_sessions)
            item_set = set(items)
            self.reference_sessions.append(items)
            self.reference_session_sets.append(item_set)
            self.reference_session_timestamps.append(timestamp)
            for item_id in item_set:
                self.item_sessions[item_id].add(session_index)

    def forward(self, interaction):
        return self.predict(interaction)

    def calculate_loss(self, interaction):
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        scores = []
        for sequence, length, item_id in zip(
            interaction[self.ITEM_SEQ],
            interaction[self.ITEM_SEQ_LEN],
            interaction[self.ITEM_ID],
        ):
            current_items = self._extract_sequence_items(sequence, int(length))
            scores.append(self._score_sequence(current_items)[int(item_id)])
        return torch.stack(scores).to(self.device)

    def full_sort_predict(self, interaction):
        batch_scores = []
        for sequence, length in zip(
            interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        ):
            current_items = self._extract_sequence_items(sequence, int(length))
            batch_scores.append(self._score_sequence(current_items))
        return torch.stack(batch_scores).to(self.device).reshape(-1)

    @staticmethod
    def _extract_sequence_items(item_seq: torch.Tensor, seq_len: int) -> list[int]:
        if seq_len <= 0:
            return []
        return [
            int(item)
            for item in item_seq[:seq_len].cpu().tolist()
            if int(item) > 0
        ]

    def _score_sequence(self, current_items: list[int]) -> torch.Tensor:
        scores = torch.zeros(self.n_items, dtype=torch.float32)
        if not current_items:
            return scores

        candidates = self._find_candidate_sessions(set(current_items))
        neighbors = []
        for session_index in candidates:
            similarity = session_similarity(
                current_items,
                self.reference_session_sets[session_index],
                self.session_weighting,
                self.similarity,
            )
            if similarity > 0:
                neighbors.append((session_index, similarity))

        neighbors.sort(key=lambda pair: (-pair[1], pair[0]))
        item_scores = score_neighbors(
            current_items,
            (
                (self.reference_session_sets[session_index], similarity)
                for session_index, similarity in neighbors[: self.neighbor_size]
            ),
            self.score_weighting,
        )
        for item_id, score in item_scores.items():
            scores[item_id] = score
        return scores

    def _find_candidate_sessions(self, current_item_set: set[int]) -> set[int]:
        candidates: set[int] = set()
        for item_id in current_item_set:
            candidates.update(self.item_sessions.get(item_id, set()))
        if self.sample_size > 0 and len(candidates) > self.sample_size:
            return set(
                sorted(
                    candidates,
                    key=lambda index: (
                        self.reference_session_timestamps[index],
                        index,
                    ),
                    reverse=True,
                )[: self.sample_size]
            )
        return candidates


# Temporary compatibility alias for existing experiment scripts/result pipelines.
VSKNNRecBole = VSKNN
