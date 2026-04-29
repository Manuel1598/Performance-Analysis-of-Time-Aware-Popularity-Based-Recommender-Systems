import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class RecentPopRecBole(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RecentPopRecBole, self).__init__(config, dataset)

        self.device = config["device"]

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.TIME_FIELD = config["TIME_FIELD"]

        self.window_days = config["window_days"]
        self.window_seconds = self.window_days * 24 * 60 * 60

        self.n_items = dataset.num(self.ITEM_ID)

        # Dummy parameter so RecBole Trainer can build an optimizer
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        self.item_popularity = torch.zeros(self.n_items, dtype=torch.float32)

        item_ids = dataset.inter_feat[self.ITEM_ID]
        timestamps = dataset.inter_feat[self.TIME_FIELD]

        # Reference point = latest timestamp in dataset
        max_timestamp = torch.max(timestamps).item()
        min_allowed_timestamp = max_timestamp - self.window_seconds

        # Count only interactions inside recent window
        for item_id, timestamp in zip(item_ids, timestamps):
            if timestamp.item() >= min_allowed_timestamp:
                self.item_popularity[item_id] += 1.0

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