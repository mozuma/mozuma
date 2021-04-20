import argparse
import json
import logging
from importlib import import_module

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.images import BaseImageDataset

logger = logging.getLogger(__name__)


def download_fun(args):
    model: BaseTorchMLModule = args.module()
    state_dict = model.get_default_pretrained_state_dict_from_provider()
    model.load_state_dict(state_dict)

    logger.info(f"Writing keys {model.state_dict().keys()}")
    with args.outfile as f:
        model.dump(f)


def run_fun(args):
    model: BaseTorchMLModule = args.module()
    file_names = [f.name for f in args.input_files]
    dataset = BaseImageDataset(file_names, args.input_files)
    indices, features = model.bulk_inference(
        dataset, tqdm_enabled=True,
        data_loader_options={"batch_size": args.batch_size, "num_workers": args.num_workers}
    )
    if hasattr(features, "tolist"):
        features = features.tolist()
    print(json.dumps(dict(zip(indices, features))))


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

    run = subparsers.add_parser('run')
    run.add_argument('--batch-size', type=int, default=None, help='Batch size for inference')
    run.add_argument('--num-workers', type=int, default=0, help='Loader number of workers')
    run.add_argument('module',
                     type=_contrib_module,
                     help='Should be in the format <module>.<MLModuleClass> '
                          'where "module" is a module in mlmodule.contrib '
                          'and "MLModuleClass" is a nn.Module')
    run.add_argument('input_files', nargs='+', type=argparse.FileType('rb'))
    run.set_defaults(func=run_fun)

    args = parser.parse_args()

    return args.func(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
