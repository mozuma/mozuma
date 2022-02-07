from typing import Union

import torch

from mlmodule.v2.base.callbacks import (
    BaseSaveBoundingBoxCallback,
    BaseSaveFeaturesCallback,
    BaseSaveLabelsCallback,
)

TorchRunnerCallbackType = Union[
    BaseSaveFeaturesCallback[torch.Tensor],
    BaseSaveLabelsCallback[torch.Tensor],
    BaseSaveBoundingBoxCallback[torch.Tensor],
]
