from io import BytesIO

import boto3
import torch
from torchvision.transforms import Compose

from mlmodule.torch.utils import torch_apply_state_to_partial_model


class TorchPretrainedModuleMixin(object):

    def get_default_pretrained_state_dict(self):
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

    def add_transforms(self, transforms):
        """Adding transforms to the list

        :param transforms:
        :return:
        """
        self.transforms += transforms

    def apply_transforms(self, x):
        """Applies the list of transforms to x

        :param x:
        :return:
        """
        return Compose(self.transforms)(x)
