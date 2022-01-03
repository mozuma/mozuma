import dataclasses
from logging import getLogger
from typing import Any, Callable, Tuple, TypeVar, Union

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from mlmodule.v2.base.runners import AbstractInferenceRunner
from mlmodule.v2.torch.results import AbstractResultsProcessor

logger = getLogger()


_DatasetType = TypeVar("_DatasetType")
_Result = TypeVar("_Result")


_BatchTypes = Union[torch.Tensor, tuple, list]


def send_batch_to_device(batch: _BatchTypes, device: torch.device) -> _BatchTypes:
    if isinstance(batch, tuple):
        return tuple(send_batch_to_device(b, device) for b in batch)
    elif isinstance(batch, list):
        return [send_batch_to_device(b, device) for b in batch]
    elif hasattr(batch, "to"):
        return batch.to(device)
    else:
        return batch


@dataclasses.dataclass
class TorchInferenceRunner(AbstractInferenceRunner[_DatasetType, Tuple[list, _Result]]):
    # Model architecture with weitghts
    model: torch.nn.Module
    # PyTorch data loader factory, builds a data loader from a dataset
    data_loader_factory: Callable[[_DatasetType], DataLoader]
    # After each batch collects and processes the results
    results_processor: AbstractResultsProcessor[Any, _Result]
    # Torch device to execute the module
    device: torch.device
    # Whether to display a tqdm progress bar
    tqdm_enabled: bool = False

    def bulk_inference(self, data: _DatasetType) -> Tuple[list, _Result]:
        # Setting model in eval mode
        self.model.eval()

        # Sending model on device
        self.model.to(self.device)

        # Disabling gradient computation
        with torch.no_grad():
            # Building data loader
            data_loader = self.data_loader_factory(data)
            # Looping through batches
            # Assume dataset is composed of tuples (item index, batch)
            n_batches = len(data_loader)
            loader = tqdm(data_loader) if self.tqdm_enabled else data_loader
            for batch_n, (indices, batch) in enumerate(loader):
                logger.debug(f"Sending batch number: {batch_n}/{n_batches}")
                # Sending data on device
                batch_on_device = send_batch_to_device(batch, self.device)
                # Running the model forward
                res = self.model(batch_on_device)
                # Processing the results
                self.results_processor.process(indices, batch, res)
                logger.debug(f"Collecting results: {batch_n}/{n_batches}")

        # Returning accumulated results
        return self.results_processor.get_results()
