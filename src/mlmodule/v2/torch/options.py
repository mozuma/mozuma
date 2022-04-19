import dataclasses
from typing import Callable, Union

import torch

# from ignite.metrics import Metric
from mlmodule.v2.torch.utils import resolve_default_torch_device


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
        loss_fn (torch.nn.Module): the loss function to use during training.
        num_epochs (int): number of epochs to train the model. Default, 24.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        metrics (dict): Ignite's metrics to compute during training.
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

    loss_fn: Union[Callable, torch.nn.Module]
    optimizer: torch.optim.Optimizer
    num_epoch: int = 24
    validate_every: int = 3
    # metrics: Dict[str, Callable[[], Metric]] = dataclasses.field(default_factory=dict)

    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False
    dist_options: dict = dataclasses.field(default_factory=dict)
    seed: int = 543
