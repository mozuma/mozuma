import argparse
import logging
from importlib import import_module

from mlmodule.torch import BaseTorchMLModule


logger = logging.getLogger(__name__)


def download_fun(args):
    model: BaseTorchMLModule = args.module()
    state_dict = model.get_default_pretrained_state_dict_from_provider()
    model.load_state_dict(state_dict)

    logger.info(f"Writing keys {model.state_dict().keys()}")
    with args.outfile as f:
        model.dump(f)


def _contrib_module(module_str):
    elements = module_str.split('.')
    if len(elements) != 2:
        raise ValueError('Format should be <module>.<MLModuleClass>')
    return getattr(import_module(f'mlmodule.contrib.{elements[0]}'), elements[1])


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='cmd', required=True)

    download = subparsers.add_parser('download')
    download.add_argument('module',
                          type=_contrib_module,
                          help='Should be in the format <module>.<MLModuleClass> '
                               'where "module" is a module in mlmodule.contrib '
                               'and "MLModuleClass" is a class that implements '
                               'the method '
                               'get_default_pretrained_state_dict_from_provider()')
    download.add_argument('outfile', type=argparse.FileType('wb'), help='Output file')
    download.set_defaults(func=download_fun)

    args = parser.parse_args()

    return args.func(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
