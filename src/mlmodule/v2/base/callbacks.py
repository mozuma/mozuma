from typing import Any, Callable, Optional, Sequence, TypeVar

from typing_extensions import Protocol

_ArrayType = TypeVar("_ArrayType", contravariant=True)


class BaseSaveFeaturesCallback(Protocol[_ArrayType]):
    def save_features(self, indices: Sequence, features: _ArrayType) -> None:
        """Callback to save features output of a module"""
        pass


class BaseSaveLabelsCallback(Protocol[_ArrayType]):
    def save_labels(self, indices: Sequence, labels_scores: _ArrayType) -> None:
        """Callback to save labels scores from a module"""
        pass


class BaseSaveBoundingBoxCallback(Protocol):

    # TODO: Bounding box type definition
    def save_bbox(self, indices: Sequence, bounding_boxes) -> None:
        """Callback to save bounding boxes from a module"""


def callbacks_caller(
    callbacks: list, callback_name: str, *args: Any, **kwds: Any
) -> None:
    """Calls all function named `callback_name` of classes in `callbacks` list with the provided arguments"""
    for c in callbacks:
        # Getting the callback function if exists
        fun: Optional[Callable] = getattr(c, callback_name, None)
        if fun:
            # Call the function
            fun(*args, **kwds)
