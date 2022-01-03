from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, List, Tuple, TypeVar, Union

from PIL.Image import Image

from mlmodule.box import BBoxOutput
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import convert_to_rgb, get_pil_image_from_file

__all__ = ("BoundingBoxDataset",)


@dataclass
class ApplyFunctionToPosition:
    """Callable that applies a given function to the element in position `pos` of a tuple
    without altering the other elements"""

    fun: Callable
    pos: int

    def __call__(self, elem: tuple) -> tuple:
        return tuple(
            list(elem[: self.pos])
            + [self.fun(elem[self.pos])]
            + list(elem[(self.pos + 1) :])
        )


IndicesType = TypeVar("IndicesType")


ReadablePathTypeList = Union[List[str], List[Path], List[IO]]


class BoundingBoxDataset(IndexedDataset[IndicesType, Tuple[Image, BBoxOutput]]):
    def __init__(
        self,
        indices: List[IndicesType],
        image_paths: ReadablePathTypeList,
        boxes: List[BBoxOutput],
        to_rgb=True,
    ):
        super().__init__(indices, list(zip(image_paths, boxes)))
        self.add_transforms(
            [ApplyFunctionToPosition(fun=get_pil_image_from_file, pos=0)]
        )
        if to_rgb:
            self.add_transforms([ApplyFunctionToPosition(fun=convert_to_rgb, pos=0)])
