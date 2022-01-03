import dataclasses
from typing import BinaryIO, Callable, Generic, List, Tuple, TypeVar

from typing_extensions import Protocol

_IndicesType = TypeVar("_IndicesType", covariant=True)
_DatasetType = TypeVar("_DatasetType", covariant=True)
_NewDatasetType = TypeVar("_NewDatasetType", covariant=True)


class TorchDataset(Protocol[_IndicesType, _DatasetType]):
    """Pytorch dataset protocol with __len__"""

    def __getitem__(self, index: int) -> Tuple[_IndicesType, _DatasetType]:
        ...

    def __len__(self) -> int:
        ...


@dataclasses.dataclass
class TorchDatasetTransformsWrapper(
    Generic[_IndicesType, _DatasetType, _NewDatasetType],
):
    dataset: TorchDataset[_IndicesType, _DatasetType]
    transform_func: Callable[[_DatasetType], _NewDatasetType]

    def __getitem__(self, index: int) -> Tuple[_IndicesType, _NewDatasetType]:
        ret_index, data = self.dataset.__getitem__(index)
        return ret_index, self.transform_func(data)

    def __len__(self) -> int:
        return self.dataset.__len__()


@dataclasses.dataclass
class ListDataset(Generic[_DatasetType]):
    objects: List[_DatasetType]

    def __getitem__(self, index: int) -> Tuple[int, _DatasetType]:
        return index, self.objects[index]

    def __len__(self) -> int:
        return len(self.objects)


@dataclasses.dataclass
class OpenBinaryFileDataset:
    paths: List[str]

    def __getitem__(self, index: int) -> Tuple[str, BinaryIO]:
        return self.paths[index], open(self.paths[index], mode="rb")

    def __len__(self) -> int:
        return len(self.paths)
