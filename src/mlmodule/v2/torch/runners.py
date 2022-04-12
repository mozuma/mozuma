import dataclasses
import logging
from typing import Any, Dict, Tuple, cast

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import manual_seed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from mlmodule.v2.base.callbacks import callbacks_caller
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.base.runners import BaseRunner
from mlmodule.v2.helpers.distributed import (
    ResultsCollector,
    register_multi_gpu_runner_logger,
)
from mlmodule.v2.torch.callbacks import TorchRunnerCallbackType
from mlmodule.v2.torch.collate import TorchMlModuleCollateFn
from mlmodule.v2.torch.datasets import (
    TorchDataset,
    TorchDatasetTransformsWrapper,
    _DatasetType,
    _IndicesType,
)
from mlmodule.v2.torch.modules import TorchMlModule
from mlmodule.v2.torch.options import TorchMultiGPURunnerOptions, TorchRunnerOptions
from mlmodule.v2.torch.utils import send_batch_to_device

logger = logging.getLogger()


def validate_data_loader_options(
    model: TorchMlModule, data_loader_options: Dict[str, Any]
) -> Dict[str, Any]:
    """Makes sure the collate function is properly defined"""
    # Making a copy of options
    data_loader_options = data_loader_options.copy()

    # Default collate function
    model_collate_fn = model.get_dataloader_collate_fn()
    if model_collate_fn:
        default_collate = TorchMlModuleCollateFn(model_collate_fn)
    else:
        default_collate = TorchMlModuleCollateFn()
    # Sets the collate_fn if not defined
    data_loader_options.setdefault("collate_fn", default_collate)

    # Checking that if set the collate_fn is an instance of TorchMlModuleCollateFn
    if not isinstance(data_loader_options["collate_fn"], TorchMlModuleCollateFn):
        logger.warning(
            "The given collate_fn is not an instance of TorchMlModuleCollateFn "
            "which could lead to type errors on callbacks"
        )

    return data_loader_options


class TorchInferenceRunner(
    BaseRunner[TorchMlModule, TorchDataset, TorchRunnerCallbackType, TorchRunnerOptions]
):
    """Runner for inference tasks on PyTorch models

    Supports CPU or single GPU inference.

    Attributes:
        model (TorchMlModule): The PyTorch model to run inference
        dataset (TorchDataset): Input dataset for the runner
        callbacks (List[TorchRunnerCallbackType]): Callbacks to save features, labels or bounding boxes
        options (TorchRunnerOptions): PyTorch options
    """

    def get_data_loader(self) -> DataLoader:
        """Creates a data loader from the options, the given dataset and the module transforms"""
        data_with_transforms = TorchDatasetTransformsWrapper(
            dataset=self.dataset,
            transform_func=transforms.Compose(self.model.get_dataset_transforms()),
        )
        return DataLoader(
            dataset=cast(Dataset, data_with_transforms),
            **validate_data_loader_options(
                self.model, self.options.data_loader_options
            ),
        )

    def apply_predictions_callbacks(
        self, indices: torch.Tensor, predictions: BatchModelPrediction[torch.Tensor]
    ) -> None:
        """Apply callback functions save_* to the returned predictions"""
        for field in dataclasses.fields(predictions):
            value = getattr(predictions, field.name)
            if value is not None:
                callbacks_caller(
                    self.callbacks, f"save_{field.name}", self.model, indices, value
                )

    def run(self) -> None:
        """Runs inference"""
        # Setting model in eval mode
        self.model.eval()

        # Sending model on device
        self.model.to(self.options.device)

        # Disabling gradient computation
        with torch.no_grad():
            # Building data loader
            data_loader = self.get_data_loader()
            # Looping through batches
            # Assume dataset is composed of tuples (item index, batch)
            n_batches = len(data_loader)
            loader = tqdm(data_loader) if self.options.tqdm_enabled else data_loader
            for batch_n, (indices, batch) in enumerate(loader):
                logger.debug(f"Sending batch number: {batch_n}/{n_batches}")
                # Sending data on device
                batch_on_device = send_batch_to_device(batch, self.options.device)
                # Running the model forward
                output = self.model(batch_on_device)
                predictions = self.model.to_predictions(output)
                # Applying callbacks on results
                self.apply_predictions_callbacks(indices, predictions)
                logger.debug(f"Collecting results: {batch_n}/{n_batches}")

        # Notify the end of the runner
        callbacks_caller(self.callbacks, "on_runner_end", self.model)


class TorchInferenceMultiGPURunner(
    BaseRunner[
        TorchMlModule, TorchDataset, TorchRunnerCallbackType, TorchMultiGPURunnerOptions
    ]
):
    """Runner for inference tasks on PyTorch models

    Supports CPU and multi-GPU inference with native torch backends.

    Attributes:
        model (TorchMlModule): The PyTorch model to run inference
        dataset (TorchDataset): Input dataset for the runner
        callbacks (List[TorchRunnerCallbackType]): Callbacks to save features, labels or bounding boxes
        options (TorchMultiGPURunnerOptions): PyTorch multi-gpu options
    """

    def get_data_loader(self) -> DataLoader:
        """Creates a data loader from the options, the given dataset and the module transforms"""
        data_with_transforms = TorchDatasetTransformsWrapper(
            dataset=self.dataset,
            transform_func=transforms.Compose(self.model.get_dataset_transforms()),
        )

        data_loader_options = validate_data_loader_options(
            self.model, self.options.data_loader_options
        )

        # Before applying, raise Ignite's logger level
        # to avoid displaying data_loader info
        if logger.level >= logging.INFO:
            logging.getLogger("ignite.distributed.auto.auto_dataloader").setLevel(
                logging.WARNING
            )

        return idist.auto_dataloader(
            dataset=cast(Dataset, data_with_transforms), **data_loader_options
        )

    def apply_predictions_callbacks(
        self, indices: torch.Tensor, predictions: BatchModelPrediction[torch.Tensor]
    ) -> None:
        """Apply callback functions save_* to the returned predictions"""
        rank = idist.get_rank()
        if rank == 0:
            for field in dataclasses.fields(predictions):
                value = getattr(predictions, field.name)
                if value is not None:
                    callbacks_caller(
                        self.callbacks, f"save_{field.name}", self.model, indices, value
                    )

    def run(self) -> None:
        """Runs inference"""
        # Some of Ignite's loggers logs a bunch of infos which we don't want
        # (to keep this runner giving the same info as the others)
        # Thus, keep them only if the current's logger level drops below INFO
        if logger.level >= logging.INFO:
            logging.getLogger("ignite.distributed.launcher.Parallel").setLevel(
                logging.WARNING
            )

        with idist.Parallel(**self.options.dist_options) as parallel:
            parallel.run(self.inference)

    def inference(self, local_rank: int) -> None:
        """Inference function to be executed in a distributed fashion"""
        rank = idist.get_rank()
        manual_seed(self.options.seed + rank)
        device = idist.device()

        # Setup dataflow
        data_loader = self.get_data_loader()

        # If data has zero size stop here, otherwise engine will raise an error
        if len(data_loader) == 0:
            logger.warning("Input data has zero size. Please provide non-empty data.")
            return

        # Adapt model for distributed settings
        if logger.level >= logging.INFO:
            logging.getLogger("ignite.distributed.auto.auto_model").setLevel(
                logging.WARNING
            )
        ddp_model = idist.auto_model(self.model)

        # Create inference engine
        inference_engine = self.create_engine(
            model=ddp_model,
            device=device,
            non_blocking=True,
        )

        # Register handler to manage predictions results
        collector = ResultsCollector(
            output_transform=self.model.to_predictions,
            callbacks_fn=self.apply_predictions_callbacks,
        )
        collector.attach(inference_engine)

        # Register handler to notify the end of the runner
        inference_engine.add_event_handler(
            Events.COMPLETED,
            lambda: callbacks_caller(self.callbacks, "on_runner_end", self.model),
        )

        # Logs basic messages, just like TorchInferenceRunner
        register_multi_gpu_runner_logger(inference_engine, data_loader, logger)

        # Setup tqdm
        if self.options.tqdm_enabled and idist.get_rank() == 0:
            pbar = ProgressBar(persist=True, bar_format=None)
            pbar.attach(inference_engine)

        inference_engine.run(data_loader)

    def create_engine(
        self,
        model: TorchMlModule,
        device: torch.device,
        non_blocking: bool = True,
    ) -> Engine:
        """Utility to create a PyTorch Ignite's Engine."""

        def inference_step(
            engine: Engine, batch_wrapper: Tuple[_IndicesType, _DatasetType]
        ):
            model.eval()
            with torch.no_grad():
                indices, batch = batch_wrapper

                # Sending data on device
                batch_on_device = send_batch_to_device(
                    batch, device=device, non_blocking=non_blocking
                )

                # Running the model forward
                output = model(batch_on_device)

                # Store data in engine's state and let the handlers
                # manage the results
                return indices, output

        engine = Engine(inference_step)

        return engine
