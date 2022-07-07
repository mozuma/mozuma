import abc
import dataclasses
from typing import Generic, List, TypeVar

_ModelType = TypeVar("_ModelType")
_DataType = TypeVar("_DataType")
_CallbackType = TypeVar("_CallbackType")
_OptionsType = TypeVar("_OptionsType")


@dataclasses.dataclass(frozen=True)
class _AbstractRunnerDataClass(
    Generic[_ModelType, _DataType, _CallbackType, _OptionsType]
):
    model: _ModelType
    dataset: _DataType
    callbacks: List[_CallbackType]
    options: _OptionsType


class _AbstractRunner(abc.ABC):
    @abc.abstractmethod
    def run(self) -> None:
        """Executes the runner"""


class BaseRunner(
    _AbstractRunner,
    _AbstractRunnerDataClass[_ModelType, _DataType, _CallbackType, _OptionsType],
):
    """A runner takes a model and run an action on it (inference, training...)

    It must implement the `run` function and call the callbacks
    to save prediction or model weights.

    Attributes:
        model (_ModelType): The model object to run the action against
        dataset (_DataType): The input dataset
        callbacks (List[_CallbackType]): Callbacks to save model state or predictions
        options (_OptionsType): Options of the runner (devices...)

    Note:
        The `_ModelType`, `_DataType`, `_CallbackType` and `_OptionsType`
        are generic types that should be specified when implementing a runner.

    Example:
        This is an example of a runner that applies a function to each element of list dataset.
        It passes the returned data to the save_features callback.

        ```python
        from mozuma.callbacks.base import (
            BaseSaveFeaturesCallback,
            callbacks_caller
        )
        from mozuma.runners import BaseRunner

        class NumpySumRunnerExample(BaseRunner[
            Callable,                   # _ModelType
            List,                       # _DataType
            BaseSaveFeaturesCallback,   # _CallbackType
            None                        # _OptionType
        ]):
            def run(self):
                for index, data in enumerate(self.dataset):
                    # Helper function to call all matching callbacks
                    callbacks_caller(
                        self.callbacks,
                        "save_features",
                        index,
                        self.model(data)
                    )
        ```
    """
