import argparse
import json
import logging
from importlib import import_module
from typing import Dict, List, Tuple, Union

import torch
from mlmodule.serializers import Serializer

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.images import ImageDataset

logger = logging.getLogger(__name__)


def download_fun(args):
    model: BaseTorchMLModule = args.module()
    state_dict = model.get_default_pretrained_state_dict_from_provider()
    model.load_state_dict(state_dict)

    logger.info(f"Writing keys {model.state_dict().keys()}")
    with args.outfile as f:
        model.dump(f)


def run_fun(args):
    model: BaseTorchMLModule = args.module(device=args.device)
    # Loading pretrained model
    model.load()
    shrink_input = None
    if hasattr(model, 'shrink_input_image_size'):
        shrink_input = model.shrink_input_image_size()
    dataset = ImageDataset(
        args.input_files,
        shrink_img_size=shrink_input
    )
    indices, features = model.bulk_inference(
        dataset, tqdm_enabled=True,
        data_loader_options={"batch_size": args.batch_size, "num_workers": args.num_workers},
        **dict(args.extra_kwargs or [])
    )
    safe_object = dict(zip(indices, Serializer(features).safe_json_types()))
    print(json.dumps(safe_object))


def _contrib_module(module_str):
    elements = module_str.split('.')
    if len(elements) != 2:
        raise ValueError('Format should be <module>.<MLModuleClass>')
    return getattr(import_module(f'mlmodule.contrib.{elements[0]}'), elements[1])


def parse_key_value_arg(cmd_values: List[str]) -> Tuple[str, Union[str, int]]:
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    try:
        (key, value) = cmd_values.split("=", 1)
    except ValueError as ex:
        raise argparse.ArgumentError(f"Argument \"{cmd_values}\" is not in k=v format")
    else:
        # Trying to parse a int otherwise leaving it as string
        try:
            value = int(value)
        except TypeError as _:
            pass
    
    return key, value


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

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

    run = subparsers.add_parser('run')
    run.add_argument('--batch-size', type=int, default=None, help='Batch size for inference')
    run.add_argument('--num-workers', type=int, default=0, help='Loader number of workers')
    run.add_argument('--device', type=torch.device, default=None, help='Torch device')
    run.add_argument('--extra-kwargs', metavar='KEY=VALUE', nargs='+', type=parse_key_value_arg)
    run.add_argument('module',
                     type=_contrib_module,
                     help='Should be in the format <module>.<MLModuleClass> '
                          'where "module" is a module in mlmodule.contrib '
                          'and "MLModuleClass" is a nn.Module')
    run.add_argument('input_files', nargs='+', help="Paths to images")
    run.set_defaults(func=run_fun)

    args = parser.parse_args()

    return args.func(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
