from io import BytesIO
from typing import Dict, Optional, List, Callable, Any, Tuple

import boto3
import torch
from torchvision.transforms import Compose

from mlmodule.torch.utils import torch_apply_state_to_partial_model


class ResizableImageInputMixin:

    def shrink_input_image_size(self) -> Tuple[int, int]:
        raise NotImplementedError()


class TorchPretrainedModuleMixin(object):

    state_dict_key: Optional[str] = None

    def get_default_pretrained_state_dict(
            self,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the state dict to apply to the current module to get a pretrained model.

        The class implementing this mixin must inherit the BaseTorchMLModule class and
        have a state_dict_key attribute, containing the key for the state dict in the
        lsir-public-assets bucket.

        :return:
        """
        s3 = boto3.resource(
            's3',
            endpoint_url="https://sos-ch-gva-2.exo.io",
            # Optionally using the provided credentials
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Select lsir-public-assets bucket
        b = s3.Bucket('lsir-public-assets')

        # Download state dict into BytesIO file
        f = BytesIO()
        b.Object(self.state_dict_key).download_fileobj(f)

        # Load the state dict
        f.seek(0)
        pretrained_state_dict = torch.load(f, map_location=lambda storage, loc: storage)
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)


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
