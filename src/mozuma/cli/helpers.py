import argparse
import functools
import json
from importlib import import_module
from typing import Any, Callable, List, Optional

import yaml

from mozuma.stores import Store


def parser_add_model_argument(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds arguments to load a model with string arguments"""
    parser.add_argument(
        "model",
        type=functools.partial(argparse_load_module, relative_to="mozuma.models"),
        help=(
            "Full path to the model class, "
            "it is considered a relative import to mozuma.models"
            "if the given path start with a '.'"
        ),
    )
    return parser


def parser_add_store_argument(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds argument to select a store class"""
    parser.add_argument(
        "--store",
        type=functools.partial(argparse_load_module, relative_to="mozuma.models"),
        default=Store,
        help=(
            "Full path to the store class, "
            "it is considered a relative import to mozuma.models"
            "if the given path start with a '.'"
        ),
    )
    return parser


def parser_add_formatter_argument(
    parser: argparse.ArgumentParser, default_formatter: Callable[[Any], str]
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--formatter",
        default=default_formatter,
        type=argparse_map_const(json=json.dumps, yaml=yaml.safe_dump),
    )
    return parser


def argparse_map_const(**mapping):
    """Maps a string to an object"""

    def parse_argument(arg):
        if arg in mapping:
            return mapping[arg]
        else:
            msg = "invalid choice: {!r} (choose from {})"
            choices = ", ".join(sorted(repr(choice) for choice in mapping.keys()))
            raise argparse.ArgumentTypeError(msg.format(arg, choices))

    return parse_argument


def argparse_load_module(object_path: str, relative_to: Optional[str] = None) -> Any:
    """Type to ArgumentParser type argument

    Will load the python object from a str path. Expected formats
        - If `object_path`=`module.submodule.Object`, it will load `from module.submodule import Object`
        - If `object_path`=`.submodule.Object`, then `relative_to` must be given and
            this will load `from {relative_to}.submodule import Object
        - If `object_path`=`module.submodule.Object(arg1, arg2, arg3)`
            it will load `from module.submodule import Object` and pre-fill the first arguments
            with strings `arg1`, `arg2`, `arg3`

    Args:
        object_path (str): The absolute or relative path to the object
        relative_to (str, optional): Will be used if `object_path` is relative.

    Returns:
        Any: The loaded object
    """
    str_arguments: Optional[List[str]] = None
    if object_path.startswith("."):
        object_path = f"{relative_to}{object_path}"
    if object_path.endswith(")"):
        object_path, raw_arguments = object_path.split("(")
        str_arguments = [a.strip() for a in raw_arguments[:-1].split(",")]

    # Importing module from string
    elements = object_path.split(".")
    if len(elements) < 2:
        raise argparse.ArgumentTypeError("Format should be <module>.<MoZuMaClass>")
    module_path = ".".join(elements[:-1])
    try:
        imported_module = getattr(import_module(module_path), elements[-1])
    except (ImportError, AttributeError):
        raise argparse.ArgumentTypeError(f"Module {object_path} cannot be imported")

    if str_arguments is not None:
        return functools.partial(imported_module, *str_arguments)
    return imported_module
