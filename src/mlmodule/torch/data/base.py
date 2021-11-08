from __future__ import annotations

import dataclasses
import logging
from torchvision.transforms import Compose
from typing import Any, Callable, Generic, Tuple, TypeVar, cast

from typing_extensions import Protocol


logger = logging.getLogger()


_IndicesType = TypeVar('_IndicesType', covariant=True)
_OutputItemsType = TypeVar('_OutputItemsType', covariant=True)


class MLModuleDatasetProtocol(Protocol[_IndicesType, _OutputItemsType]):
    """Pytorch dataset protocol with __len__"""

    def __getitem__(self, index: int) -> Tuple[_IndicesType, _OutputItemsType]:
        ...

    def __len__(self) -> int:
        ...


@dataclasses.dataclass
class IndexedDataset(Generic[_IndicesType, _OutputItemsType]):
    """Torch dataset returning a tuple of indices and data point"""
    # Indices to identify items
    indices: list[_IndicesType]
    # Actual data to pass to transforms
    items: list

    transforms: list = dataclasses.field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.items) != len(self.indices):
            raise ValueError("Inconsistent length between indices and items")

    def add_transforms(self, transforms: list[Callable]) -> None:
        """Adding transforms to the list

        :param transforms:
        :return:
        """
        self.transforms += transforms

    def apply_transforms(self, x: Any) -> _OutputItemsType:
        """Applies the list of transforms to x

        :param x:
        :return:
        """
        return Compose(self.transforms)(x)

    def __getitem__(self, item_num: int) -> tuple[_IndicesType, _OutputItemsType]:
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
