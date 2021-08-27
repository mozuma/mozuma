from io import BytesIO
from typing import Dict, Optional, List, Callable, Any, Tuple

import boto3
import torch
from torchvision.transforms import Compose

from mlmodule.torch.utils import torch_apply_state_to_partial_model


class ResizableImageInputMixin:

    def shrink_input_image_size(self) -> Tuple[int, int]:
        raise NotImplementedError()


class TorchDatasetTransformsMixin:

    transforms: List[Callable]

    def add_transforms(self, transforms: List[Callable]) -> None:
        """Adding transforms to the list

        :param transforms:
        :return:
        """
        self.transforms += transforms

    def apply_transforms(self, x: Any) -> Any:
        """Applies the list of transforms to x

        :param x:
        :return:
        """
        return Compose(self.transforms)(x)


class DownloadPretrainedStateFromProvider:

    def get_default_pretrained_state_dict_from_provider(self) -> Dict[str, torch.Tensor]:
        """Allows to download pretrained state dir from model provider directly (used in the cli download)"""
        raise NotImplementedError()
