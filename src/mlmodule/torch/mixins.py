from io import BytesIO
from typing import Dict, Optional, List, Callable, Any

import boto3
import torch
from torchvision.transforms import Compose

from mlmodule.torch.utils import torch_apply_state_to_partial_model


class TorchPretrainedModuleMixin(object):

    state_dict_key = None

    def get_default_pretrained_state_dict(
            self: torch.nn.Module,
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
        s3 = boto3.resource('s3', endpoint_url="https://sos-ch-gva-2.exo.io")
        # Select lsir-public-assets bucket
        b = s3.Bucket('lsir-public-assets')

        # Download state dict into BytesIO file
        f = BytesIO()
        b.Object(self.state_dict_key).download_fileobj(f)

        # Load the state dict
        f.seek(0)
        pretrained_state_dict = torch.load(f, map_location=lambda storage, loc: storage)
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)


class TorchDatasetTransformsMixin(object):

    transforms = None

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
