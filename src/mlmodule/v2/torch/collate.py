import dataclasses
from typing import Callable, List, TypeVar

from torch.utils.data.dataloader import default_collate

_IndicesType = TypeVar("_IndicesType")
_TargetType = TypeVar("_TargetType")


@dataclasses.dataclass
class TorchMlModuleCollateFn:
    collate_fn: Callable = default_collate

    def __call__(self, data):
        """Collate function that leaves the indices untouched and applies the default to other slots of the tuple"""
        indices: List[_IndicesType] = []
        dataset_payload: List[tuple] = []
        for d in data:
            indices.append(d[0])
            dataset_payload.append(d[1])
        return (indices, self.collate_fn(dataset_payload))


@dataclasses.dataclass
class TorchMlModuleTrainingCollateFn:
    collate_fn: Callable = default_collate

    def __call__(self, data):
        """Collate function that leaves the targets untouched and applies the default to other slots of the tuple"""
        dataset_payload: List[tuple] = []
        targets: List[_TargetType] = []
        for d in data:
            dataset_payload.append(d[0])
            targets.append(d[1])
        return (self.collate_fn(dataset_payload), targets)
