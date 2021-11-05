import abc
from typing import Any, Generic, TypeVar
from typing_extensions import Protocol


_Input = TypeVar("_Input")
_Model = TypeVar("_Model")
_Result = TypeVar("_Result")


class AbstractRunner(abc.ABC):
    """A runner takes a model and run an action on it (inference, training...)"""


class AbstractInferenceRunner(AbstractRunner, Generic[_Input, _Result]):
    """A runner for model inference"""

    @abc.abstractmethod
    def bulk_inference(self, data: _Input) -> _Result:
        """Runs inference on the given data"""


_Runner = TypeVar("_Runner", covariant=True, bound=AbstractRunner)


class RunnerFactory(Protocol, Generic[_Model, _Runner]):
    model: _Model
    options: Any

    def get_runner(self) -> _Runner:
        ...
