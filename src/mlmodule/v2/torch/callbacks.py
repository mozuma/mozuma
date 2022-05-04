from typing import Union

import torch

from mlmodule.callbacks.base import (
    BaseSaveBoundingBoxCallback,
    BaseSaveFeaturesCallback,
    BaseSaveLabelsCallback,
    BaseSaveVideoFramesCallback,
)

TorchRunnerCallbackType = Union[
    BaseSaveFeaturesCallback[torch.Tensor],
    BaseSaveLabelsCallback[torch.Tensor],
    BaseSaveVideoFramesCallback[torch.Tensor],
    BaseSaveBoundingBoxCallback,
]
