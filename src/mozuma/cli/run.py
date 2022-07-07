import argparse
import dataclasses
import json
import pathlib
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from typing_extensions import Literal

from mozuma.callbacks.memory import (
    CollectBoundingBoxesInMemory,
    CollectFeaturesInMemory,
    CollectLabelsInMemory,
    CollectVideoFramesInMemory,
)
from mozuma.cli.helpers import (
    parser_add_formatter_argument,
    parser_add_model_argument,
    parser_add_store_argument,
)
from mozuma.cli.types import ArgMoZuMaOptions, CLICommandDefinition, CLIObjectFactory
from mozuma.predictions import BatchModelPrediction
from mozuma.predictions.serializers import batch_model_prediction_to_dict
from mozuma.states import StateKey
from mozuma.stores import Store
from mozuma.stores.abstract import AbstractStateStore
from mozuma.torch.datasets import ImageDataset, LocalBinaryFilesDataset, TorchDataset
from mozuma.torch.modules import TorchModel
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.torch.utils import resolve_default_torch_device

_FileType = Literal["im", "vi"]


@dataclasses.dataclass
class ArgMoZuMaTorchRunOptions(ArgMoZuMaOptions):
    model: CLIObjectFactory[TorchModel]
    file_names: Sequence[pathlib.Path]
    device: torch.device
    training_id: Optional[str] = None
    store: Callable[[], AbstractStateStore] = dataclasses.field(default=Store)
    batch_size: Optional[int] = None
    num_workers: int = 0
    file_type: _FileType = "im"
    formatter: Callable[[Union[dict, Sequence]], str] = json.dumps

    def instantiate_model(self) -> TorchModel:
        return self.model()


def parse_run_arguments(parser: argparse.ArgumentParser):
    parser_add_model_argument(parser)
    parser_add_store_argument(parser)
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Loader number of workers"
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=resolve_default_torch_device(),
        help="Torch device",
    )
    parser.add_argument(
        "--file-type",
        choices=("im", "vi"),
        default="im",
        help="The type of input files",
    )
    parser.add_argument("--training-id", help="The training id of the model to load")
    parser.add_argument(
        "file_names", nargs="+", type=pathlib.Path, help="Paths to files"
    )
    parser_add_formatter_argument(parser, json.dumps)


def _unique_training_id(store: AbstractStateStore, model: TorchModel) -> str:
    training_ids = [s.training_id for s in store.get_state_keys(model.state_type)]
    if len(training_ids) > 1:
        raise ValueError(
            f"Ambiguous training id, please specify --trainind-id [{','.join(training_ids)}]"
        )
    elif len(training_ids) == 0:
        raise ValueError(f"No matching compatible states saved for model {model}")

    return training_ids[0]


def _get_dataset(
    file_names: Sequence[pathlib.Path], file_type: _FileType
) -> TorchDataset:
    if file_type == "im":
        return ImageDataset(LocalBinaryFilesDataset(file_names))
    elif file_type == "vi":
        return LocalBinaryFilesDataset(file_names)
    else:
        raise ValueError(f"Unknown type for input : {file_type}")


def _merge_predictions(
    features: CollectFeaturesInMemory,
    labels: CollectLabelsInMemory,
    frames: CollectVideoFramesInMemory,
    bounding_boxes: CollectBoundingBoxesInMemory,
) -> Tuple[list, BatchModelPrediction[np.ndarray]]:
    collected_items = [
        (features, "features"),
        (labels, "label_scores"),
        (frames, "frames"),
        (bounding_boxes, "bounding_boxes"),
    ]
    c_indices: Optional[list] = None
    c_batch_predictions = BatchModelPrediction[np.ndarray]()
    for collect, key in collected_items:
        indices: list = collect.indices  # type: ignore
        if len(indices) == 0:
            # Skip, there are no data in this objects
            continue

        # Getting new ordering
        ordering = np.argsort(indices)
        ordered_indices = [indices[i] for i in ordering]
        if c_indices is None:
            # This is the first non-empty object
            c_indices = ordered_indices

        # Adding the predictions
        predictions = getattr(collect, key)
        if isinstance(predictions, np.ndarray):
            ordered_predictions = predictions[ordering]
        else:
            ordered_predictions = [predictions[i] for i in ordering]

        # Add values in order
        setattr(c_batch_predictions, key, ordered_predictions)

    if c_indices is None:
        raise ValueError("No data in the given collect objects")

    return c_indices, c_batch_predictions


def torch_run(options: ArgMoZuMaTorchRunOptions) -> None:
    """Runs a TorchModel from CLI"""
    # Building the model
    model = options.instantiate_model()

    # Loading model weights
    store = options.store()
    training_id = options.training_id or _unique_training_id(store, model)
    store.load(model, StateKey(state_type=model.state_type, training_id=training_id))

    # Getting dataset
    dataset = _get_dataset(file_names=options.file_names, file_type=options.file_type)

    # Callbacks and runner
    callbacks: list = [
        CollectFeaturesInMemory(),
        CollectLabelsInMemory(),
        CollectVideoFramesInMemory(),
        CollectBoundingBoxesInMemory(),
    ]

    runner = TorchInferenceRunner(
        model=model,
        dataset=dataset,
        callbacks=callbacks,
        options=TorchRunnerOptions(
            device=options.device,
            data_loader_options={
                "batch_size": options.batch_size,
                "num_workers": options.num_workers,
            },
        ),
    )
    runner.run()

    # Merging callbacks data
    indices, predictions = _merge_predictions(*callbacks)

    print(
        options.formatter(
            batch_model_prediction_to_dict([str(i) for i in indices], predictions)
        )
    )


COMMAND = CLICommandDefinition(
    name="run",
    help_text="Run a model against a list of files",
    args_parser=parse_run_arguments,
    command_fun=torch_run,
    options_class=ArgMoZuMaTorchRunOptions,
)
