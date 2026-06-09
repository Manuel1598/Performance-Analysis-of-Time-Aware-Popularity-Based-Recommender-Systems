from collections import defaultdict
import math

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class VSKNNRecBole(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VSKNNRecBole, self).__init__(config, dataset)

        self.device = config["device"]
        self.time_field = config["TIME_FIELD"]

        self.k = config["vsknn_k"]
        self.sample_size = config["vsknn_sample_size"]
        self.popularity_weight = self._get_config_float(
            config=config,
            key="vsknn_popularity_weight",
            default=0.0,
        )

        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.reference_sessions: list[list[int]] = []
        self.reference_session_sets: list[set[int]] = []
        self.reference_session_timestamps: list[float] = []
        self.item_sessions: dict[int, set[int]] = defaultdict(set)
        self.item_popularity = self._compute_item_popularity(dataset)

        self._build_reference_sessions(dataset)

    @staticmethod
    def _get_config_float(config, key: str, default: float) -> float:
        try:
            return float(config[key])
        except KeyError:
            return default

    def _compute_item_popularity(self, dataset) -> torch.Tensor:
        item_ids = dataset.inter_feat[self.ITEM_ID].long()
        item_counts = torch.bincount(
            item_ids,
            minlength=self.n_items,
        ).float()

        interaction_count = max(float(len(item_ids)), 1.0)
        item_popularity = item_counts / interaction_count

        return item_popularity.clamp_min(1.0 / interaction_count)

    def _build_reference_sessions(self, dataset) -> None:
        item_seq_data = dataset.inter_feat[self.ITEM_SEQ]
        item_seq_len_data = dataset.inter_feat[self.ITEM_SEQ_LEN]
        target_items = dataset.inter_feat[self.ITEM_ID]
        timestamps = dataset.inter_feat[self.time_field]

        for row_idx in range(len(item_seq_data)):
            seq_len = int(item_seq_len_data[row_idx])
            item_seq = item_seq_data[row_idx][:seq_len].cpu().tolist()

            target_item = int(target_items[row_idx])

            # Historical sequence + target item as reference session
            session_items = [int(item_id) for item_id in item_seq if int(item_id) > 0]

            if target_item > 0:
                session_items.append(target_item)

            if not session_items:
                continue

            session_index = len(self.reference_sessions)

            self.reference_sessions.append(session_items)
            self.reference_session_timestamps.append(float(timestamps[row_idx]))
            session_item_set = set(session_items)
            self.reference_session_sets.append(session_item_set)

            for item_id in session_item_set:
                self.item_sessions[item_id].add(session_index)

    def forward(self, interaction):
        return self.predict(interaction)

    def calculate_loss(self, interaction):
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_item = interaction[self.ITEM_ID]

        scores = []

        for seq, seq_len, item_id in zip(item_seq, item_seq_len, target_item):
            current_items = self._extract_sequence_items(seq, int(seq_len))
            session_scores = self._score_sequence(current_items)
            scores.append(session_scores[int(item_id)])

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        batch_scores = []

        for seq, seq_len in zip(item_seq, item_seq_len):
            current_items = self._extract_sequence_items(seq, int(seq_len))
            scores = self._score_sequence(current_items)
            batch_scores.append(scores)

        scores_tensor = torch.stack(batch_scores, dim=0).to(self.device)

        return scores_tensor.reshape(-1)

    def _extract_sequence_items(self, item_seq: torch.Tensor, seq_len: int) -> list[int]:
        if seq_len <= 0:
            return []

        items = item_seq[:seq_len].cpu().tolist()
        return [int(item_id) for item_id in items if int(item_id) > 0]

    def _score_sequence(self, current_items: list[int]) -> torch.Tensor:
        scores = torch.zeros(self.n_items, dtype=torch.float32)

        if not current_items:
            return scores

        current_item_set = set(current_items)

        candidate_sessions = self._find_candidate_sessions(current_item_set)

        if not candidate_sessions:
            return scores

        similarities = []

        for session_index in candidate_sessions:
            similarity = self._cosine_similarity(
                current_item_set=current_item_set,
                session_index=session_index,
            )

            if similarity > 0:
                similarities.append((session_index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[: self.k]

        for session_index, similarity in nearest_neighbors:
            neighbor_items = self.reference_sessions[session_index]

            for item_id in neighbor_items:
                scores[item_id] += self._apply_popularity_weight(
                    item_id=item_id,
                    score=similarity,
                )

        return scores

    def _apply_popularity_weight(self, item_id: int, score: float) -> float:
        if self.popularity_weight <= 0.0:
            return score

        popularity = float(self.item_popularity[item_id].item())
        return score / (popularity ** self.popularity_weight)

    def _find_candidate_sessions(self, current_item_set: set[int]) -> set[int]:
        candidate_sessions = set()

        for item_id in current_item_set:
            candidate_sessions.update(self.item_sessions.get(item_id, set()))

        if self.sample_size > 0 and len(candidate_sessions) > self.sample_size:
            candidate_sessions = set(
                sorted(
                    candidate_sessions,
                    key=lambda session_index: (
                        self.reference_session_timestamps[session_index],
                        session_index,
                    ),
                    reverse=True,
                )[: self.sample_size]
            )

        return candidate_sessions

    def _cosine_similarity(
        self,
        current_item_set: set[int],
        session_index: int,
    ) -> float:
        neighbor_item_set = self.reference_session_sets[session_index]

        if not neighbor_item_set:
            return 0.0

        intersection_size = len(current_item_set.intersection(neighbor_item_set))

        if intersection_size == 0:
            return 0.0

        denominator = math.sqrt(len(current_item_set) * len(neighbor_item_set))

        return intersection_size / denominator
