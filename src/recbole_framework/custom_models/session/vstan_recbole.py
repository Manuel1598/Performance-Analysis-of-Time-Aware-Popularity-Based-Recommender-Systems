from collections import defaultdict
import math
import random

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class VSTANRecBole(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VSTANRecBole, self).__init__(config, dataset)

        self.device = config["device"]
        self.seed = config["seed"]

        self.k = config["vstan_k"]
        self.sample_size = config["vstan_sample_size"]
        self.position_decay = config["vstan_position_decay"]
        self.idf_weighting = config["vstan_idf_weighting"]

        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.reference_sessions: list[list[int]] = []
        self.reference_session_sets: list[set[int]] = []
        self.item_sessions: dict[int, set[int]] = defaultdict(set)
        self.item_idf: dict[int, float] = {}

        self._build_reference_sessions(dataset)
        self._compute_idf_weights()

    def _build_reference_sessions(self, dataset) -> None:
        item_seq_data = dataset.inter_feat[self.ITEM_SEQ]
        item_seq_len_data = dataset.inter_feat[self.ITEM_SEQ_LEN]
        target_items = dataset.inter_feat[self.ITEM_ID]

        for row_idx in range(len(item_seq_data)):
            seq_len = int(item_seq_len_data[row_idx])
            item_seq = item_seq_data[row_idx][:seq_len].cpu().tolist()
            target_item = int(target_items[row_idx])

            session_items = [int(item_id) for item_id in item_seq if int(item_id) > 0]

            if target_item > 0:
                session_items.append(target_item)

            if not session_items:
                continue

            session_index = len(self.reference_sessions)

            self.reference_sessions.append(session_items)

            session_item_set = set(session_items)
            self.reference_session_sets.append(session_item_set)

            for item_id in session_item_set:
                self.item_sessions[item_id].add(session_index)

    def _compute_idf_weights(self) -> None:
        number_of_sessions = len(self.reference_sessions)

        if number_of_sessions == 0:
            return

        for item_id, sessions in self.item_sessions.items():
            document_frequency = len(sessions)
            self.item_idf[item_id] = math.log(
                (number_of_sessions + 1) / (document_frequency + 1)
            ) + 1.0

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
            similarity = self._weighted_session_similarity(
                current_items=current_items,
                current_item_set=current_item_set,
                session_index=session_index,
            )

            if similarity > 0:
                similarities.append((session_index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[: self.k]

        for session_index, similarity in nearest_neighbors:
            neighbor_items = self.reference_sessions[session_index]

            for position, item_id in enumerate(neighbor_items):
                item_score = similarity

                # Items later in the neighbor session receive higher weight
                item_score *= self._position_weight(
                    position=position,
                    session_length=len(neighbor_items),
                )

                if self.idf_weighting:
                    item_score *= self.item_idf.get(item_id, 1.0)

                scores[item_id] += item_score

        return scores

    def _find_candidate_sessions(self, current_item_set: set[int]) -> set[int]:
        candidate_sessions = set()

        for item_id in current_item_set:
            candidate_sessions.update(self.item_sessions.get(item_id, set()))

        if self.sample_size > 0 and len(candidate_sessions) > self.sample_size:
            seed_key = f"{self.seed}:{','.join(map(str, sorted(current_item_set)))}"
            rng = random.Random(seed_key)
            candidate_sessions = set(
                rng.sample(sorted(candidate_sessions), self.sample_size)
            )

        return candidate_sessions

    def _weighted_session_similarity(
        self,
        current_items: list[int],
        current_item_set: set[int],
        session_index: int,
    ) -> float:
        neighbor_items = self.reference_sessions[session_index]
        neighbor_item_set = self.reference_session_sets[session_index]

        if not neighbor_item_set:
            return 0.0

        similarity_sum = 0.0

        for position, item_id in enumerate(current_items):
            if item_id not in neighbor_item_set:
                continue

            weight = self._position_weight(
                position=position,
                session_length=len(current_items),
            )

            if self.idf_weighting:
                weight *= self.item_idf.get(item_id, 1.0)

            similarity_sum += weight

        if similarity_sum == 0.0:
            return 0.0

        denominator = math.sqrt(len(current_item_set) * len(neighbor_item_set))
        return similarity_sum / denominator

    def _position_weight(self, position: int, session_length: int) -> float:
        if session_length <= 1:
            return 1.0

        distance_from_end = session_length - position - 1
        return math.exp(-self.position_decay * distance_from_end)
