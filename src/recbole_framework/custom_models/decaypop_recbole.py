import math
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class DecayPopRecBole(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DecayPopRecBole, self).__init__(config, dataset)

        self.device = config["device"]

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.TIME_FIELD = config["TIME_FIELD"]

        self.decay_lambda = config["decay_lambda"]

        self.n_items = dataset.num(self.ITEM_ID)

        # Dummy parameter so RecBole can build an optimizer
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.item_popularity = torch.zeros(self.n_items, dtype=torch.float32)

        item_ids = dataset.inter_feat[self.ITEM_ID]
        timestamps = dataset.inter_feat[self.TIME_FIELD]

        # Reference point = latest timestamp in dataset
        max_timestamp = torch.max(timestamps).item()

        for item_id, timestamp in zip(item_ids, timestamps):
            delta_t = max_timestamp - timestamp.item()
            weight = math.exp(-self.decay_lambda * delta_t)
            self.item_popularity[item_id] += weight

        self.item_popularity = self.item_popularity.to(self.device)

    def forward(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def calculate_loss(self, interaction):
        # Non-trainable model, but RecBole expects a differentiable loss tensor
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_size = user.shape[0]

        scores = self.item_popularity.unsqueeze(0).repeat(batch_size, 1)

        # RecBole expects flattened output
        return scores.view(-1)