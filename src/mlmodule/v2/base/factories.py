import abc
from typing import Any, Generic, Optional, TypeVar

from mlmodule.v2.base.models import AbstractModelStore, MLModuleModelStore
from mlmodule.v2.base.runners import AbstractRunner

_Model = TypeVar("_Model")
_Runner = TypeVar("_Runner", covariant=True, bound=AbstractRunner)


class AbstractRunnerFactory(abc.ABC, Generic[_Model, _Runner]):
    """Factory to create a runner instance

    The goal is to bundle all the required components to run inference / training on a model:
    - The model code

    The factory should accepts additional parameters to change behaviour of
    - Model state management
    - Runner execution options
    """

    model_store: Optional[AbstractModelStore]
    options: Any

    @abc.abstractmethod
    def get_runner(self) -> _Runner:
        """Returns an instance of a runner"""

    def get_model_store(self) -> AbstractModelStore:
        """Returns the model store object to load and save model state"""
        return self.model_store or MLModuleModelStore()

    @abc.abstractmethod
    def get_model(self) -> _Model:
        """Return the MLModule model with preloaded state if available"""
