import dataclasses
import logging
from typing import TypeVar, List, Tuple, cast
from mlmodule.torch.base import MLModuleDatasetProtocol

from mlmodule.torch.mixins import TorchDatasetTransformsMixin


logger = logging.getLogger(__name__)


_IndicesType = TypeVar('_IndicesType')
_OutputItemsType = TypeVar('_OutputItemsType')


@dataclasses.dataclass
class IndexedDataset(MLModuleDatasetProtocol[_IndicesType, _OutputItemsType], TorchDatasetTransformsMixin):
    """Torch dataset returning a tuple of indices and data point"""
    # Indices to identify items
    indices: List[_IndicesType]
    # Actual data to pass to transforms
    items: list

    transforms: list = dataclasses.field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.items) != len(self.indices):
            raise ValueError("Inconsistent length between indices and items")

    def __getitem__(self, item_num: int) -> Tuple[_IndicesType, _OutputItemsType]:
        index = self.indices[item_num]
        value = self.items[item_num]
        logger.debug(f"Reading item {item_num}, index: {index}")

        value = self.apply_transforms(value)
        if type(value) == tuple:
            ret = (index, *value)
        else:
            ret = index, value
        return cast(Tuple[_IndicesType, _OutputItemsType], ret)

    def __len__(self) -> int:
        return len(self.indices)
