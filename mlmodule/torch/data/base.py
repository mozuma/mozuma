import logging

from torch.utils.data.dataset import Dataset

from mlmodule.torch.mixins import TorchDatasetTransformsMixin


logger = logging.getLogger(__name__)


class BaseIndexedDataset(Dataset, TorchDatasetTransformsMixin):

    def __init__(self, item_list):
        """
        :param item_list: Must be a string
        """
        self.item_list = item_list
        self.transforms = []

    def __getitem__(self, item):
        logger.debug(f"Reading item {item} -> {self.item_list[item]}")
        return item, self.apply_transforms(self.item_list[item])

    def __len__(self):
        return len(self.item_list)
