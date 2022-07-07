from typing import Union

import torch

from mozuma.callbacks.base import (
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
