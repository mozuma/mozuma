import dataclasses
import logging

import torch

from mlmodule.v2.torch.collate import TorchMlModuleCollateFn
from mlmodule.v2.torch.utils import resolve_default_torch_device

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TorchRunnerOptions:
    device: torch.device = dataclasses.field(
        default_factory=resolve_default_torch_device
    )
    data_loader_options: dict = dataclasses.field(default_factory=dict)
    tqdm_enabled: bool = False

    def __post_init__(self):
        """Sets the collate_fn if not defined"""
        self.data_loader_options.setdefault("collate_fn", TorchMlModuleCollateFn())

        # Checking that if set the collate_fn is an instance of TorchMlModuleCollateFn
        if not isinstance(
            self.data_loader_options["collate_fn"], TorchMlModuleCollateFn
        ):
            logger.warning(
                "The given collate_fn is not an instance of TorchMlModuleCollateFn "
                "which could lead to type errors on callbacks"
            )
