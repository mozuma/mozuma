import dataclasses
from logging import getLogger
from typing import cast

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from mlmodule.v2.base.callbacks import callbacks_caller
from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.base.runners import BaseRunner
from mlmodule.v2.torch.callbacks import TorchRunnerCallbackType
from mlmodule.v2.torch.datasets import TorchDataset, TorchDatasetTransformsWrapper
from mlmodule.v2.torch.modules import TorchMlModule
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.utils import send_batch_to_device

logger = getLogger()


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
            **self.options.data_loader_options,
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
                predictions = self.model.forward_predictions(batch_on_device)
                # Applying callbacks on results
                self.apply_predictions_callbacks(indices, predictions)
                logger.debug(f"Collecting results: {batch_n}/{n_batches}")
