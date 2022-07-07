import dataclasses
from typing import Callable, List, TypeVar

from torch.utils.data.dataloader import default_collate

_IndicesType = TypeVar("_IndicesType")


@dataclasses.dataclass
class TorchModelCollateFn:
    collate_fn: Callable = default_collate

    def __call__(self, data):
        """Collate function that leaves the indices untouched and applies the default to other slots of the tuple"""
        indices: List[_IndicesType] = []
        dataset_payload: List[tuple] = []
        for d in data:
            indices.append(d[0])
            dataset_payload.append(d[1])
        return (indices, self.collate_fn(dataset_payload))
