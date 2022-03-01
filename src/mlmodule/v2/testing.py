import dataclasses
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

import torch

from mlmodule.v2.stores.abstract import AbstractStateStore

_Module = TypeVar("_Module")


@dataclasses.dataclass
class ModuleTestConfiguration(Generic[_Module]):
    """Identifies a MLModule configuration for generic tests"""

    module_class: Type[_Module]
    module_args: List = dataclasses.field(default_factory=list)
    module_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    has_state: bool = True
    # Is it a Pytorch model
    is_pytorch: bool = True
    # The shape of the input tensor to forward
    batch_input_shape: Optional[List[int]] = None
    batch_input_type: Callable = torch.Tensor
    # Model provider store
    provider_store: Optional[AbstractStateStore] = None
    provider_store_training_ids: Set[str] = dataclasses.field(default_factory=set)

    def get_module(self) -> _Module:
        return self.module_class(*self.module_args, **self.module_kwargs)

    def __repr__(self) -> str:
        attributes: list = (
            [self.module_class.__name__]
            + self.module_args
            + list(self.module_kwargs.values())
        )
        return "-".join(str(a) for a in attributes)
