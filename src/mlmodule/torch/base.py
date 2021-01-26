from mlmodule.base import BaseMLModule
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from mlmodule.torch.utils import generic_inference


class BaseTorchMLModule(BaseMLModule, nn.Module):

    def __init__(self, device=None):
        super().__init__()
        self.device = device or self._resolve_device()

    @classmethod
    def _resolve_device(cls):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load(self, fp=None):
        """Loads model from file or from a default pretrained model if `fp=None`

        :param fp:
        :param load_options:
        :return:
        """

        # Getting state dict
        if fp:
            fp.seek(0)
            state = torch.load(fp)
        else:
            # Getting default pretrained state dict
            # Requires TorchPretrainedModuleMixin to be implemented
            state = self.get_default_pretrained_state_dict()

        # Loading state
        self.load_state_dict(state)

        return self

    def dump(self, fp):
        torch.save(self.state_dict(), fp)

    def get_data_loader(self, data, **data_loader_options):
        """Configured data loader with applied transforms

        :param data:
        :param data_loader_options:
        :return:
        """
        # Adding module transforms
        data.add_transforms(self.get_dataset_transforms())
        # Data loader default options
        data_loader_options.setdefault("shuffle", False)
        data_loader_options.setdefault("drop_last", False)
        # We send to pin memory only if using CUDA device
        data_loader_options.setdefault("pin_memory", self.device != torch.device('cpu'))
        # Building data loader
        return DataLoader(data, **data_loader_options)

    @classmethod
    def get_results_handler(cls):
        """Runs after the forward pass at inference

        :return:
        """
        return zip

    def get_forward_func(self):
        return self.__call__

    def bulk_inference(self, data, **data_loader_options):
        """Run the model against all elements in data

        :type data: Dataset, TorchDatasetTransformsMixin
        :param data:
        :return:
        """
        loader = self.get_data_loader(data, **data_loader_options)
        # Running inference batch loop
        return generic_inference(self, loader, self.get_forward_func(), self.get_results_handler(), self.device)

    def get_dataset_transforms(self):
        """Returns callable that transform the input data before the forward pass

        :return: List of transforms
        """
        return []
