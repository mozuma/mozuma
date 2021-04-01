import logging
from typing import Generic, TypeVar, List, Tuple

from torch.utils.data.dataset import Dataset

from mlmodule.torch.mixins import TorchDatasetTransformsMixin


logger = logging.getLogger(__name__)


TI = TypeVar('TI')
TD = TypeVar('TD')


class IndexedDataset(Dataset, TorchDatasetTransformsMixin, Generic[TI, TD]):

    def __init__(self, indices: List[TI], items: List[TD]):
        """
        :param indices: Indices to identify items
        :param items: Actual data
        """
        if len(items) != len(indices):
            raise ValueError("Inconsistent length between indices and items")
        self.indices = indices
        self.items = items
        self.transforms = []

    def __getitem__(self, item_num: int) -> Tuple[TI, TD]:
        index = self.indices[item_num]
        value = self.items[item_num]
        logger.debug(f"Reading item {item_num}, index: {index}")

        return index, self.apply_transforms(value)

    def __len__(self):
        return len(self.indices)
