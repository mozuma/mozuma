import dataclasses
import logging

import torch

from mlmodule.v2.torch.utils import resolve_default_torch_device

logger = logging.getLogger(__name__)


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
        tqdm_enabled (bool): Whether to print a `tqdm` progress bar
        seed (int): random state seed to set. Default, 543.
        dist_options (dict): Options passed to `ignite.distributed.Parallel`.

    Note:
        `data_loader_options`'s options `batch_size` and `num_worker`
        will be automatically scaled according to `world_size` and `nprocs` respectively.
        For more info visit [`auto_dataloader` documentation](https://pytorch.org/ignite/v0.4.8/generated/ignite.distributed.auto.auto_dataloader.html#ignite.distributed.auto.auto_dataloader).

    Note:
        `dist_options` usually include `backend` and `nproc_per_node` parameters, sometimes also `start_method`.
        For more info check [PyTorch Ignite's distributed documentation](https://pytorch.org/ignite/v0.4.8/distributed.html).
    """  # noqa: E501

    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False
    seed: int = 543
    dist_options: dict = dataclasses.field(default_factory=dict)
