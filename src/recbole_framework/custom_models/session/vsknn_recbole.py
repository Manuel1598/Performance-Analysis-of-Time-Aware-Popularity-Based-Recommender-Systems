from collections import defaultdict
import math
import random

import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class VSKNNRecBole(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VSKNNRecBole, self).__init__(config, dataset)

        self.device = config["device"]

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.TIME_FIELD = config["TIME_FIELD"]

        self.n_items = dataset.num(self.ITEM_ID)

        self.k = config["vsknn_k"]
        self.sample_size = config["vsknn_sample_size"]

        # Dummy parameter so RecBole Trainer can build an optimizer
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.session_items: dict[int, list[int]] = defaultdict(list)
        self.item_sessions: dict[int, set[int]] = defaultdict(set)

        user_ids = dataset.inter_feat[self.USER_ID].cpu().tolist()
        item_ids = dataset.inter_feat[self.ITEM_ID].cpu().tolist()
        timestamps = dataset.inter_feat[self.TIME_FIELD].cpu().tolist()

        interactions = list(zip(user_ids, item_ids, timestamps))
        interactions.sort(key=lambda x: (x[0], x[2]))

        for session_id, item_id, _ in interactions:
            self.session_items[int(session_id)].append(int(item_id))
            self.item_sessions[int(item_id)].add(int(session_id))

        self.session_lengths = {
            session_id: len(items)
            for session_id, items in self.session_items.items()
        }

    def forward(self, interaction):
        return self.predict(interaction)

    def calculate_loss(self, interaction):
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        scores = []

        for session_id, target_item in zip(user.cpu().tolist(), item.cpu().tolist()):
            session_scores = self._score_session(int(session_id))
            scores.append(session_scores[int(target_item)])

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().tolist()

        batch_scores = []

        for session_id in users:
            scores = self._score_session(int(session_id))
            batch_scores.append(scores)

        scores_tensor = torch.stack(batch_scores, dim=0).to(self.device)

        return scores_tensor.reshape(-1)

    def _score_session(self, session_id: int) -> torch.Tensor:
        current_items = self.session_items.get(session_id, [])

        scores = torch.zeros(self.n_items, dtype=torch.float32)

        if not current_items:
            return scores

        current_item_set = set(current_items)

        candidate_sessions = self._find_candidate_sessions(current_item_set, session_id)

        if not candidate_sessions:
            return scores

        similarities = []

        for neighbor_session_id in candidate_sessions:
            similarity = self._cosine_similarity(
                current_item_set=current_item_set,
                neighbor_session_id=neighbor_session_id,
            )

            if similarity > 0:
                similarities.append((neighbor_session_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[: self.k]

        for neighbor_session_id, similarity in nearest_neighbors:
            neighbor_items = self.session_items[neighbor_session_id]

            for item_id in neighbor_items:
                if item_id in current_item_set:
                    continue

                scores[item_id] += similarity

        return scores

    def _find_candidate_sessions(
        self,
        current_item_set: set[int],
        current_session_id: int,
    ) -> set[int]:
        candidate_sessions = set()

        for item_id in current_item_set:
            candidate_sessions.update(self.item_sessions.get(item_id, set()))

        candidate_sessions.discard(current_session_id)

        if self.sample_size > 0 and len(candidate_sessions) > self.sample_size:
            candidate_sessions = set(
                random.sample(list(candidate_sessions), self.sample_size)
            )

        return candidate_sessions

    def _cosine_similarity(
        self,
        current_item_set: set[int],
        neighbor_session_id: int,
    ) -> float:
        neighbor_items = self.session_items.get(neighbor_session_id, [])
        neighbor_item_set = set(neighbor_items)

        if not neighbor_item_set:
            return 0.0

        intersection_size = len(current_item_set.intersection(neighbor_item_set))

        if intersection_size == 0:
            return 0.0

        denominator = math.sqrt(len(current_item_set) * len(neighbor_item_set))

        return intersection_size / denominator