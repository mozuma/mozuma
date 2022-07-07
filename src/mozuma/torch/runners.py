import dataclasses
import logging
from typing import Any, Callable, Dict, Tuple, Union, cast

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (
    Engine,
    Events,
    EventsList,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.metrics import Metric, RunningAverage
from ignite.utils import manual_seed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from mozuma.callbacks.base import BaseRunnerEndCallback, callbacks_caller
from mozuma.callbacks.states import SaveModelState
from mozuma.helpers.ignite import ResultsCollector, register_multi_gpu_runner_logger
from mozuma.predictions import BatchModelPrediction
from mozuma.runners import BaseRunner
from mozuma.torch.callbacks import TorchRunnerCallbackType
from mozuma.torch.collate import TorchModelCollateFn
from mozuma.torch.datasets import (
    TorchDataset,
    TorchDatasetTransformsWrapper,
    TorchTrainingDataset,
)
from mozuma.torch.modules import TorchModel
from mozuma.torch.options import (
    TorchMultiGPURunnerOptions,
    TorchRunnerOptions,
    TorchTrainingOptions,
)
from mozuma.torch.utils import (
    disable_ignite_logger,
    log_evaluation_metrics,
    prepare_batch_for_training,
    send_batch_to_device,
)

logger = logging.getLogger(__name__)


def validate_data_loader_options(
    model: TorchModel, data_loader_options: Dict[str, Any]
) -> Dict[str, Any]:
    """Makes sure the collate function is properly defined"""
    # Making a copy of options
    data_loader_options = data_loader_options.copy()

    # Default collate function
    model_collate_fn = model.get_dataloader_collate_fn()
    if model_collate_fn:
        default_collate = TorchModelCollateFn(model_collate_fn)
    else:
        default_collate = TorchModelCollateFn()
    # Sets the collate_fn if not defined
    data_loader_options.setdefault("collate_fn", default_collate)

    # Checking that if set the collate_fn is an instance of TorchModelCollateFn
    if not isinstance(data_loader_options["collate_fn"], TorchModelCollateFn):
        logger.warning(
            "The given collate_fn is not an instance of TorchModelCollateFn "
            "which could lead to type errors on callbacks"
        )

    return data_loader_options


class TorchInferenceRunner(
    BaseRunner[TorchModel, TorchDataset, TorchRunnerCallbackType, TorchRunnerOptions]
):
    """Runner for inference tasks on PyTorch models

    Supports CPU or single GPU inference.

    Attributes:
        model (TorchModel): The PyTorch model to run inference
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
        TorchModel, TorchDataset, TorchRunnerCallbackType, TorchMultiGPURunnerOptions
    ]
):
    """Runner for inference tasks on PyTorch models

    Supports CPU and multi-GPU inference with native torch backends.

    Attributes:
        model (TorchModel): The PyTorch model to run inference
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

        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.auto.auto_dataloader", logger)

        return idist.auto_dataloader(
            dataset=cast(Dataset, data_with_transforms), **data_loader_options
        )  # type: ignore

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
        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.launcher.Parallel", logger)

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

        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.auto.auto_model", logger)

        # Adapt model for distributed settings
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
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(inference_engine)

        inference_engine.run(data_loader)

    def create_engine(
        self,
        model: torch.nn.Module,
        device: torch.device,
        non_blocking: bool = True,
    ) -> Engine:
        """Utility to create a PyTorch Ignite's Engine."""

        def inference_step(
            engine: Engine,
            batch_wrapper: Tuple,
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


@dataclasses.dataclass(frozen=True)
class TorchTrainingRunner(
    BaseRunner[
        TorchModel,
        Tuple[TorchTrainingDataset, TorchTrainingDataset],
        Union[BaseRunnerEndCallback, SaveModelState],
        TorchTrainingOptions,
    ]
):
    """Runner for training tasks on PyTorch models

    Supports CPU and multi-GPU training with multiple backends.

    Attributes:
        model (TorchModel): The PyTorch model to run inference
        dataset (Tuple[TorchTrainingDataset, TorchTrainingDataset]): Train and test datasets for the runner
        callbacks (List[Union[BaseRunnerEndCallback, SaveModelState]]):
            Callback to save model weights plus one or more callbacks for when the runner ends.
        options (TorchTrainingOptions): PyTorch training options
    """

    def __post_init__(self) -> None:
        #  Warn user if model isn't trainable
        if not getattr(self.model, "is_trainable", None):
            logger.warning(self.model.__class__.__name__ + " is not trainable!")

    def _apply_transforms_on_training_data(self, data: Any) -> Tuple:
        """This functions allows to use TorchTrainingDataset with TorchDatasetTransformsWrapper.
        It apply dataset transforms to the payload leaving the targets untouched."""
        payload, target = data

        return (
            transforms.Compose(self.model.get_dataset_transforms())(payload),
            target,
        )

    def get_data_loader(self, dataset: TorchTrainingDataset, **kwargs) -> DataLoader:
        """Creates the data loaders from the options, the given datasets and the module transforms.
        The first data loader will be used to train, se second to test.
        """
        data_with_transforms = TorchDatasetTransformsWrapper(
            dataset=dataset,
            transform_func=self._apply_transforms_on_training_data,
        )

        data_loader_options = self.options.data_loader_options.copy()
        data_loader_options.update(kwargs)

        data_loader_options = validate_data_loader_options(
            self.model, data_loader_options
        )

        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.auto.auto_dataloader", logger)

        return idist.auto_dataloader(
            dataset=cast(Dataset, data_with_transforms), **data_loader_options
        )  # type: ignore

    def run(self) -> None:
        """Runs training"""
        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.launcher.Parallel", logger)

        with idist.Parallel(**self.options.dist_options) as parallel:
            parallel.run(self.training)

    def training(self, local_rank: int) -> None:
        """Training function to be executed in a distributed fashion"""
        rank = idist.get_rank()
        manual_seed(self.options.seed + rank)
        device = idist.device()

        train_loader = self.get_data_loader(self.dataset[0])
        test_loader = self.get_data_loader(self.dataset[1], shuffle=False)

        # If data has zero size stop here, otherwise engine will raise an error
        if len(train_loader) == 0 or len(test_loader) == 0:
            logger.warning("Input data has zero size. Please provide non-empty data.")
            return

        # Disable Ignite logging
        disable_ignite_logger("ignite.distributed.auto.auto_model", logger)

        # Adapt model for distributed settings
        ddp_model = idist.auto_model(self.model)

        # Adapt optimizer for distributed settings
        optimizer = idist.auto_optim(self.options.optimizer)

        criterion = self.options.criterion
        if isinstance(self.options.criterion, torch.nn.Module):
            criterion = self.options.criterion.to(device)

        # Create trainer for current task
        trainer = self.create_trainer(
            model=ddp_model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            non_blocking=True,
        )

        # Setup evaluator engine to perform model's validation and compute metrics
        train_evaluator = self.create_evaluator(
            model=ddp_model,
            metrics=self.options.metrics,
            device=device,
            non_blocking=True,
        )
        evaluator = self.create_evaluator(
            model=ddp_model,
            metrics=self.options.metrics,
            device=device,
            non_blocking=True,
        )

        # Attach additional loggers
        if self.options.loggers_factory is not None:
            self.options.loggers_factory(trainer, train_evaluator, evaluator)

        @trainer.on(
            Events.EPOCH_COMPLETED(every=self.options.validate_every) | Events.COMPLETED
        )
        def run_validation(engine: Engine) -> None:
            epoch = trainer.state.epoch

            state = train_evaluator.run(train_loader)
            if idist.get_rank() == 0:
                log_evaluation_metrics(
                    logger,
                    epoch,
                    state.times["COMPLETED"] or -1,
                    "Train",
                    state.metrics,
                )

            state = evaluator.run(test_loader)
            if idist.get_rank() == 0:
                log_evaluation_metrics(
                    logger,
                    epoch,
                    state.times["COMPLETED"] or -1,
                    "Test",
                    state.metrics,
                )

        trainer.run(train_loader, max_epochs=self.options.num_epoch)

    def create_trainer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: torch.device,
        non_blocking: bool = True,
    ) -> Engine:
        """Utility to create a PyTorch Ignite's engine for training."""

        trainer = create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch_for_training,
        )

        # Register handler to save model weights
        # Note: there should be only one callback of such
        checkpoint_events = EventsList()
        checkpoint_events |= Events.COMPLETED
        if self.options.checkpoint_every:
            checkpoint_events |= Events.EPOCH_COMPLETED(
                every=self.options.checkpoint_every
            )
        trainer.add_event_handler(
            checkpoint_events,
            lambda engine: callbacks_caller(
                self.callbacks, "save_model_state", engine, self.model
            ),
        )

        # Register handler to notify the end of the runner
        trainer.add_event_handler(
            Events.COMPLETED,
            lambda: callbacks_caller(self.callbacks, "on_runner_end", self.model),
        )

        # Compute average of the loss and attach it to trainer
        RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")

        # Setup tqdm
        if self.options.tqdm_enabled and idist.get_rank() == 0:
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(trainer, metric_names=["avg_loss"])

        return trainer

    def create_evaluator(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Metric],
        device: torch.device,
        non_blocking: bool = True,
    ) -> Engine:
        """Utility to create a PyTorch Ignite's engine for evaluation."""
        evaluator = create_supervised_evaluator(
            model=model,
            metrics=metrics,
            device=device,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch_for_training,
        )

        for name, metric in self.options.metrics.items():
            metric.attach(evaluator, name)

        # Setup tqdm
        if self.options.tqdm_enabled and idist.get_rank() == 0:
            pbar = ProgressBar(desc="Evaluation ", persist=False)
            pbar.attach(evaluator)

        return evaluator
