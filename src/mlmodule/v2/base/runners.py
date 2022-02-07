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
    """A runner takes a model and run an action on it (inference, training...)"""

    pass
