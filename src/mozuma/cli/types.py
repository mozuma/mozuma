import dataclasses
from argparse import ArgumentParser
from typing import Callable, Generic, Type, TypeVar

from typing_extensions import Protocol

_ObjClass = TypeVar("_ObjClass", covariant=True)
_CLIOptions = TypeVar("_CLIOptions")


@dataclasses.dataclass
class ArgMoZuMaOptions:
    pass


class CLIObjectFactory(Protocol[_ObjClass]):
    """A factory that constructs an object from str arguments only"""

    def __call__(self, *args: str) -> _ObjClass:
        """Should only be called with str arguments as we don't perform casting on CLI arguments"""


@dataclasses.dataclass
class CLICommandDefinition(Generic[_CLIOptions]):
    name: str
    help_text: str
    args_parser: Callable[[ArgumentParser], None]
    command_fun: Callable[[_CLIOptions], None]
    options_class: Type[_CLIOptions]
