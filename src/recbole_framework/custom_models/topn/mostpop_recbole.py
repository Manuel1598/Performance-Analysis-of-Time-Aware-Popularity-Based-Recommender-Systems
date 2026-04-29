import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class MostPopRecBole(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MostPopRecBole, self).__init__(config, dataset)

        self.device = config["device"]
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]

        self.n_items = dataset.num(self.ITEM_ID)

        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        item_ids = dataset.inter_feat[self.ITEM_ID].long()

        self.item_popularity = torch.bincount(
            item_ids,
            minlength=self.n_items,
        ).float().to(self.device)

    def forward(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def calculate_loss(self, interaction):
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_size = user.shape[0]

        scores = self.item_popularity.unsqueeze(0).expand(batch_size, -1)
        return scores.reshape(-1)