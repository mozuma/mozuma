import dataclasses
from typing import Callable, Generic, Optional, TypeVar

from mozuma.stores.abstract import AbstractStateStore

_Module = TypeVar("_Module")


@dataclasses.dataclass
class ModuleTestConfiguration(Generic[_Module]):
    """Identifies a MoZuMa configuration for generic tests"""

    name: str
    module_factory: Callable[[], _Module]
    training_id: Optional[str] = None
    has_state: bool = True
    # Is it a Pytorch model
    is_pytorch: bool = True
    # The shape of the input tensor to forward
    batch_factory: Optional[Callable] = None
    # Model provider store
    provider_store: Optional[AbstractStateStore] = None

    def get_module(self) -> _Module:
        return self.module_factory()

    def __repr__(self) -> str:
        return self.name
