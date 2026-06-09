import math

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class BaseSessionPopularityRecBole(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BaseSessionPopularityRecBole, self).__init__(config, dataset)

        self.device = config["device"]
        self.item_field = config["ITEM_ID_FIELD"]
        self.n_items = dataset.num(self.item_field)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.item_popularity = torch.zeros(
            self.n_items,
            dtype=torch.float32,
            device=self.device,
        )

    def forward(self, interaction):
        return self.predict(interaction)

    def calculate_loss(self, interaction):
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        batch_size = item_seq.shape[0]

        scores = self.item_popularity.unsqueeze(0).repeat(batch_size, 1)
        scores[:, 0] = -float("inf")

        return scores.view(-1)


class SessionMostPopRecBole(BaseSessionPopularityRecBole):
    def __init__(self, config, dataset):
        super(SessionMostPopRecBole, self).__init__(config, dataset)

        item_ids = dataset.inter_feat[self.item_field].long()
        self.item_popularity = torch.bincount(
            item_ids,
            minlength=self.n_items,
        ).float().to(self.device)


class SessionRecentPopRecBole(BaseSessionPopularityRecBole):
    def __init__(self, config, dataset):
        super(SessionRecentPopRecBole, self).__init__(config, dataset)

        self.time_field = config["TIME_FIELD"]
        window_seconds = config["window_days"] * 24 * 60 * 60

        item_ids = dataset.inter_feat[self.item_field].long()
        timestamps = dataset.inter_feat[self.time_field]

        max_timestamp = torch.max(timestamps).item()
        min_timestamp = max_timestamp - window_seconds
        recent_mask = timestamps >= min_timestamp

        recent_item_ids = item_ids[recent_mask]
        self.item_popularity = torch.bincount(
            recent_item_ids,
            minlength=self.n_items,
        ).float().to(self.device)


class SessionDecayPopRecBole(BaseSessionPopularityRecBole):
    def __init__(self, config, dataset):
        super(SessionDecayPopRecBole, self).__init__(config, dataset)

        self.time_field = config["TIME_FIELD"]
        decay_lambda = config["decay_lambda"]

        item_ids = dataset.inter_feat[self.item_field].long()
        timestamps = dataset.inter_feat[self.time_field]
        max_timestamp = torch.max(timestamps).item()

        item_popularity = torch.zeros(self.n_items, dtype=torch.float32)

        for item_id, timestamp in zip(item_ids, timestamps):
            delta_t = max_timestamp - timestamp.item()
            item_popularity[item_id] += math.exp(-decay_lambda * delta_t)

        self.item_popularity = item_popularity.to(self.device)
