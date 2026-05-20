import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class MostPopRecBole(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MostPopRecBole, self).__init__(config, dataset)


        self.device = config["device"]

        # RecBole field names
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]

        # Number of items in the RecBole dataset
        self.n_items = dataset.num(self.ITEM_ID)

        # Dummy parameter so RecBole can build an optimizer
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        # Popularity scores for all item ids in RecBole's internal index space
        self.item_popularity = torch.zeros(self.n_items, dtype=torch.float32)

        # InterFeat contains the observed interactions
        item_ids = dataset.inter_feat[self.ITEM_ID]

        # Count item occurrences
        item_ids = dataset.inter_feat[self.ITEM_ID].long()
        self.item_popularity = torch.bincount(
            item_ids,
            minlength=self.n_items
        ).float()
        self.item_popularity = self.item_popularity.to(self.device)

        self.item_popularity = self.item_popularity.to(self.device)

        print("DEBUG MostPop nonzero items:", torch.count_nonzero(self.item_popularity).item())
        print("DEBUG MostPop max popularity:", torch.max(self.item_popularity).item())
        print("DEBUG MostPop top items:", torch.topk(self.item_popularity, k=10).indices.tolist())

    def forward(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def calculate_loss(self, interaction):
        # MostPop is non-trainable, but RecBole expects a loss tensor
        return self.dummy_param.sum() * 0.0

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        return self.item_popularity[item]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch_size = user.shape[0]

        scores = self.item_popularity.unsqueeze(0).repeat(batch_size, 1)

        # Padding-Item sicher ausschließen
        scores[:, 0] = -float("inf")

        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print("DEBUG MostPop top item ids:", torch.topk(scores[0], k=10).indices.tolist())
            print("DEBUG MostPop top scores:", torch.topk(scores[0], k=10).values.tolist())

        return scores.view(-1)

