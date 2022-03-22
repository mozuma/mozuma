import dataclasses
import logging
from collections import abc
from typing import Any, Dict, Tuple, cast

import ignite.distributed as idist
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ignite.contrib.handlers import ProgressBar
from ignite.distributed.utils import one_rank_only
from ignite.engine import Engine, Events
from ignite.utils import manual_seed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from mlmodule.v2.base.callbacks import callbacks_caller
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.base.runners import BaseRunner
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

    @staticmethod
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

    def get_data_loader(self) -> DataLoader:
        """Creates a data loader from the options, the given dataset and the module transforms"""
        data_with_transforms = TorchDatasetTransformsWrapper(
            dataset=self.dataset,
            transform_func=transforms.Compose(self.model.get_dataset_transforms()),
        )
        return DataLoader(
            dataset=cast(Dataset, data_with_transforms),
            **TorchInferenceRunner.validate_data_loader_options(
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
    _queue: mp.Queue = None
    _is_reduced: bool = False

    def get_data_loader(self) -> DataLoader:
        """Creates a data loader from the options, the given dataset and the module transforms"""
        data_with_transforms = TorchDatasetTransformsWrapper(
            dataset=self.dataset,
            transform_func=transforms.Compose(self.model.get_dataset_transforms()),
        )

        data_loader_options = TorchInferenceRunner.validate_data_loader_options(
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
        for field in dataclasses.fields(predictions):
            value = getattr(predictions, field.name)
            if value is not None:
                if not isinstance(value, abc.Iterator):
                    value = [value]

                # Pytorch's DataParallel returns a mapping from all model outputs.
                # If som apply the callback each time
                for val in value:
                    callbacks_caller(
                        self.callbacks,
                        f"save_{field.name}",
                        self.model,
                        indices,
                        val,
                    )

    def apply_collected_predictions_results(self) -> None:
        """Apply predictions results computed for a subset of the initial dataset,
        such as when spawning multiple subprocesses"""
        # Collect indices and predictions from the queue, sent by group (sub)process with rank 0
        while not self._queue.empty():
            indices, predictions = self._queue.get()

            # Each message is from a (sub)process, within a group, and should contain
            # indices and predictions for all the callbacks
            if len(indices) != len(predictions) == len(self.callbacks):
                logger.warning(
                    "There a problem with the length of indices and predictions"
                )
                continue

            # Apply the callback 'save_*' function
            for c in self.callbacks:
                callbacks_caller(
                    [c],
                    f"save_{c.PREDICTION_TYPE}",
                    self.model,
                    indices.pop(0) if indices else indices,
                    predictions.pop(0) if predictions else predictions,
                )

    def gather_predictions_results(self, engine: Engine) -> None:
        """Gather predictions results among (sub)processes in a group,
        then from rank 0 send them in the internal queue
        """
        ws = idist.get_world_size()

        # Gather indices andz predictions from all (sub)processes
        # in the group and for all callbacks
        if ws > 1 and not self._is_reduced:
            _gather_objects = [None for _ in range(0, ws)]
            _indices = [c.indices for c in self.callbacks]
            dist.all_gather_object(_gather_objects, _indices)
            _indices = _gather_objects

            _gather_objects = [None for _ in range(0, ws)]
            _predictions = [getattr(c, c.PREDICTION_TYPE, None) for c in self.callbacks]
            dist.all_gather_object(_gather_objects, _predictions)
            _predictions = _gather_objects

        self._is_reduced = True

        # From (sub)process with rank 0, send results to the queue
        # so the master process can fetch them later
        if idist.get_rank() == 0:
            for idx, preds in zip(_indices, _predictions):
                self._queue.put((idx, preds))

    def run(self) -> None:
        """Runs inference"""
        # Some of Ignite's loggers logs a bunch of infos which we don't want
        # (to keep this runner giving the same info as the others)
        # Thus, keep them only if the current's logger level drops below INFO
        if logger.level >= logging.INFO:
            logging.getLogger("ignite.distributed.launcher.Parallel").setLevel(
                logging.WARNING
            )

        # If backend is enabled, setup queue for master<-processes comunication
        backend_setting = self.options.dist_options.get("backend", None)
        if backend_setting:
            manager = mp.Manager()
            self._queue = manager.Queue()

        with idist.Parallel(**self.options.dist_options) as parallel:
            parallel.run(self.inference)

        # If backend is enabled, apply collected results from sub-processes
        if backend_setting:
            logger.debug("Apply results from sub-processes callbacks")
            self.apply_collected_predictions_results()

        # Notify the end of the runner
        callbacks_caller(self.callbacks, "on_runner_end", self.model)

    def inference(self, local_rank: int) -> None:
        """Inference function to be executed in a distributed fashion"""
        rank = idist.get_rank()
        manual_seed(self.options.seed + rank)
        device = idist.device()

        # Setup dataflow
        data_loader = self.get_data_loader()

        # Adapt model for distributed settings
        if logger.level >= logging.INFO:
            logging.getLogger("ignite.distributed.auto.auto_model").setLevel(
                logging.WARNING
            )
        ddp_model = idist.auto_model(self.model)

        # Create inference engine (similar to an evaluator)
        inference_engine = self.create_engine(
            model=ddp_model,
            data_loader=data_loader,
            # metrics=,
            device=device,
            non_blocking=True,
        )

        inference_engine.run(data_loader)

    def create_engine(
        self,
        model: TorchMlModule,
        data_loader: DataLoader,
        device: torch.device,
        non_blocking: bool = False,
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
                # predictions = model.forward_predictions(batch_on_device)
                predictions = model(batch_on_device)

                # Applying callbacks on results
                self.apply_predictions_callbacks(indices, predictions)

                # Predictions are stored in the callbacks,
                # thus there's no need to return something here (= engine state)
                return

        engine = Engine(inference_step)

        # Gather prediction results
        ws = idist.get_world_size()
        if ws > 1:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, self.gather_predictions_results
            )

        # Logs basic messages, just like TorchInferenceRunner
        @engine.on(Events.EPOCH_STARTED)
        @one_rank_only()
        def on_start(engine):
            engine.state.n_batches = len(data_loader)

        @engine.on(Events.ITERATION_STARTED)
        @one_rank_only()
        def on_itertation_started(engine):
            s = engine.state
            logger.debug(f"Sending batch number: {s.iteration}/{s.n_batches}")

        @engine.on(Events.ITERATION_COMPLETED)
        @one_rank_only()
        def on_itertation_completed(engine):
            s = engine.state
            logger.debug(f"Collecting results: {s.iteration}/{s.n_batches}")

        if self.options.tqdm_enabled and idist.get_rank() == 0:
            pbar = ProgressBar(persist=True, bar_format=None)
            pbar.attach(engine)

        return engine
