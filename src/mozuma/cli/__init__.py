import argparse
from typing import List

from mozuma.cli import checks, ls, run
from mozuma.cli.types import CLICommandDefinition

COMMAND_DEFINITIONS: List[CLICommandDefinition] = [
    ls.COMMAND,
    run.COMMAND,
    checks.COMMAND,
]


def cli():
    parser = argparse.ArgumentParser("mozuma", description="CLI to list and run models")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for cmd_def in COMMAND_DEFINITIONS:
        # Configuring command parser
        cmd_parser = subparsers.add_parser(cmd_def.name, help=cmd_def.help_text)
        cmd_def.args_parser(cmd_parser)
        cmd_parser.set_defaults(func=cmd_def.command_fun)
        cmd_parser.set_defaults(options_class=cmd_def.options_class)

    args = parser.parse_args()
    args.func(
        args.options_class(
            **{
                k: v
                for k, v in vars(args).items()
                if k not in ("func", "options_class", "cmd")
            }
        )
    )
