import dataclasses
from argparse import ArgumentParser
from typing import Callable, Sequence, Union

import yaml

from mlmodule.cli.helpers import (
    parser_add_formatter_argument,
    parser_add_model_argument,
    parser_add_store_argument,
)
from mlmodule.cli.types import (
    ArgMlModuleOptions,
    CLICommandDefinition,
    CLIObjectFactory,
)
from mlmodule.models.types import ModelWithState
from mlmodule.v2.stores import Store
from mlmodule.v2.stores.abstract import AbstractStateStore


@dataclasses.dataclass
class ArgMlModuleListStatesOptions(ArgMlModuleOptions):
    model: CLIObjectFactory[ModelWithState]
    model_args: Sequence[str] = dataclasses.field(default_factory=tuple)
    store: Callable[[], AbstractStateStore] = dataclasses.field(default=Store)
    formatter: Callable[[Union[dict, Sequence]], str] = yaml.safe_dump

    def instantiate_model(self) -> ModelWithState:
        return self.model(*self.model_args)


def parse_ls_arguments(
    parser: ArgumentParser,
):
    parser_add_model_argument(parser)
    parser_add_store_argument(parser)
    parser_add_formatter_argument(parser, yaml.safe_dump)


def ls_fun(options: ArgMlModuleListStatesOptions) -> None:
    model = options.instantiate_model()
    state_keys = options.store().get_state_keys(model.state_type)
    dict_state_keys = [dataclasses.asdict(sk) for sk in state_keys]
    print(options.formatter(dict_state_keys))


COMMAND = CLICommandDefinition(
    name="ls",
    help_text="List available states for a model",
    args_parser=parse_ls_arguments,
    command_fun=ls_fun,
    options_class=ArgMlModuleListStatesOptions,
)
