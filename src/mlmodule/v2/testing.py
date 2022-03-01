import dataclasses
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

_Module = TypeVar("_Module")


@dataclasses.dataclass
class ModuleTestConfiguration(Generic[_Module]):
    """Identifies a MLModule configuration for generic tests"""

    module_class: Type[_Module]
    module_args: List = dataclasses.field(default_factory=list)
    module_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    has_state: bool = True
    is_pytorch: bool = True
    batch_input_shape: Optional[List[int]] = None

    def get_module(self) -> _Module:
        return self.module_class(*self.module_args, **self.module_kwargs)

    def __repr__(self) -> str:
        attributes: list = (
            [self.module_class.__name__]
            + self.module_args
            + list(self.module_kwargs.values())
        )
        return "-".join(str(a) for a in attributes)
