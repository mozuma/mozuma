import abc
from typing import Generic, TypeVar


_Input = TypeVar("_Input")
_Result = TypeVar("_Result")


class AbstractRunner(abc.ABC):
    """A runner takes a model and run an action on it (inference, training...)"""


class AbstractInferenceRunner(AbstractRunner, Generic[_Input, _Result]):
    """A runner for model inference"""

    @abc.abstractmethod
    def bulk_inference(self, data: _Input) -> _Result:
        """Runs inference on the given data"""
