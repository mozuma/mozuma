import dataclasses
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

import torch

from mlmodule.v2.stores.abstract import AbstractStateStore

_Module = TypeVar("_Module")


@dataclasses.dataclass
class ModuleTestConfiguration(Generic[_Module]):
    """Identifies a MLModule configuration for generic tests"""

    name: str
    module_factory: Callable[[], _Module]
    has_state: bool = True
    # Is it a Pytorch model
    is_pytorch: bool = True
    # The shape of the input tensor to forward
    batch_factory: Optional[Callable] = None
    # Model provider store
    provider_store: Optional[AbstractStateStore] = None
    provider_store_training_ids: Set[str] = dataclasses.field(default_factory=set)

    def get_module(self) -> _Module:
        return self.module_factory()

    def __repr__(self) -> str:
        return self.name
