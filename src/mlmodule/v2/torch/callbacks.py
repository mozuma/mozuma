from typing import Union

import torch

from mlmodule.v2.base.callbacks import (
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
