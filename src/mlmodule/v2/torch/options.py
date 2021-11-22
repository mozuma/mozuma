import dataclasses

import torch

from mlmodule.v2.torch.utils import resolve_default_torch_device


@dataclasses.dataclass(frozen=True)
class TorchRunnerOptions:
    device: torch.device = dataclasses.field(default_factory=resolve_default_torch_device)
    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False
