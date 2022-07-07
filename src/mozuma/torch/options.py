import dataclasses
from typing import Callable, Dict, Optional, Union

import dill
import torch
from ignite.engine import Engine
from ignite.metrics import Metric

from mozuma.torch.utils import resolve_default_torch_device


@dataclasses.dataclass(frozen=True)
class TorchRunnerOptions:
    """Options for PyTorch runners

    Attributes:
        device (torch.device): Torch device
        data_loader_options (dict): Options passed to `torch.utils.dataloader.DataLoader`.
        tqdm_enabled (bool): Whether to print a `tqdm` progress bar
    """

    device: torch.device = dataclasses.field(
        default_factory=resolve_default_torch_device
    )
    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False


@dataclasses.dataclass(frozen=True)
class TorchMultiGPURunnerOptions:
    """Options for PyTorch multi-gpu runners

    Attributes:
        data_loader_options (dict): Options passed to `torch.utils.dataloader.DataLoader`.
        tqdm_enabled (bool): Whether to print a `tqdm` progress bar. Default, False.
        dist_options (dict): Options passed to `ignite.distributed.Parallel`.
        seed (int): random state seed to set. Default, 543.

    Note:
        `data_loader_options`'s options `batch_size` and `num_worker`
        will be automatically scaled according to `world_size` and `nprocs` respectively.
        For more info visit [`auto_dataloader` documentation](https://pytorch.org/ignite/v0.4.8/generated/ignite.distributed.auto.auto_dataloader.html).

    Note:
        `dist_options` usually include `backend` and `nproc_per_node` parameters.
        For more info visit [PyTorch Ignite's distributed documentation](https://pytorch.org/ignite/distributed.html).
    """  # noqa: E501

    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False
    dist_options: dict = dataclasses.field(default_factory=dict)
    seed: int = 543


@dataclasses.dataclass(frozen=True)
class TorchTrainingOptions:
    """Options for PyTorch training runners

    Attributes:
        criterion (Union[Callable, torch.nn.Module]): the loss function to use during training.
        optimizer (torch.optim.Optimizer): Optimization strategy to use during training.
        num_epochs (int): number of epochs to train the model.
        validate_every (int): run model's validation every ``validate_every`` epochs.
        checkpoint_every (Optional[int]): store training checkpoint every ``checkpoint_every`` epochs.
        metrics (Dict[str, Metric]): Dictionary where values are Ignite's metrics to compute during evaluation.
        data_loader_options (dict): Options passed to `torch.utils.dataloader.DataLoader`.
        tqdm_enabled (bool): Whether to print a `tqdm` progress bar. Default, False.
        dist_options (dict): Options passed to `ignite.distributed.Parallel`.
        seed (int): random state seed to set. Default, 543.
        loggers_factory (Callable[[Engine, Engine, Engine], None] | None): Function to attach additional loggers
            to training runner and its evaluators internal (PyTorch Ignite) engines.
            The function receives three engines, one for training and two for evaluation, where
            the first is for the train set and second for the test set.

    Note:
        `data_loader_options`'s options `batch_size` and `num_worker`
        will be automatically scaled according to `world_size` and `nprocs` respectively.
        For more info visit [`auto_dataloader` documentation](https://pytorch.org/ignite/v0.4.8/generated/ignite.distributed.auto.auto_dataloader.html).

    Note:
        `dist_options` usually include `backend` and `nproc_per_node` parameters.
        For more info visit [PyTorch Ignite's distributed documentation](https://pytorch.org/ignite/distributed.html).
    """  # noqa: E501

    criterion: Union[Callable, torch.nn.Module]
    optimizer: torch.optim.Optimizer
    num_epoch: int
    validate_every: int
    metrics: Dict[str, Metric] = dataclasses.field(default_factory=dict)
    checkpoint_every: Optional[int] = None

    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False
    dist_options: dict = dataclasses.field(default_factory=dict)
    seed: int = 543
    loggers_factory: Optional[Callable[[Engine, Engine, Engine], None]] = None

    def __getstate__(self):
        # This method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()

        # Don't pickle special fields
        del state["metrics"]
        del state["loggers_factory"]

        # Instead, pickle them manually with dill
        state["pickled-metrics"] = dill.dumps(self.metrics)
        state["pickled-loggers"] = dill.dumps(self.loggers_factory)

        return state

    def __setstate__(self, state):
        # Put pre-pickled fields aside
        pickled_metrics = state.pop("pickled-metrics", None)
        pickled_logger = state.pop("pickled-loggers", None)

        # Restore all other fields
        self.__dict__.update(state)

        # Unpickle the special fields with dill
        object.__setattr__(self, "metrics", dill.loads(pickled_metrics))
        object.__setattr__(self, "loggers_factory", dill.loads(pickled_logger))
