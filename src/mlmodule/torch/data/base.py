import logging

from torch.utils.data.dataset import Dataset

from mlmodule.torch.mixins import TorchDatasetTransformsMixin


logger = logging.getLogger(__name__)


class IndexedDataset(Dataset, TorchDatasetTransformsMixin):

    def __init__(self, indices, items):
        """
        :param indices: Indices to identify items
        :param items: Actual data
        """
        if len(items) != len(indices):
            raise ValueError("Inconsistent length between indices and items")
        self.indices = indices
        self.items = items
        self.transforms = []

    def __getitem__(self, item_num):
        index = self.indices[item_num]
        value = self.items[item_num]
        logger.debug(f"Reading item {item_num}, index: {index}")

        return index, self.apply_transforms(value)

    def __len__(self):
        return len(self.indices)
