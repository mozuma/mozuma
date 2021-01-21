import multiprocessing

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

    def load(self, fp=None, **load_options):
        """Loads model from file or from a default pretrained model if `fp=None`

        :param fp:
        :param load_options:
        :return:
        """
        load_options["map_location"] = load_options.get("map_location", self.device)

        # Getting state dict
        if fp:
            fp.seek(0)
            state = torch.load(fp, **load_options)
        else:
            # Getting default pretrained state dict
            # Requires TorchPretrainedModuleMixin to be implemented
            state = self.get_default_pretrained_state_dict(**load_options)

        # Loading state
        self.load_state_dict(state)

        return self

    def dump(self, fp):
        torch.save(self.state_dict(), fp)

    def bulk_inference(self, data, **data_loader_options):
        """Run the model against all elements in data

        :type data: Dataset, TorchDatasetTransformsMixin
        :param batch_size:
        :param num_workers:
        :param data:
        :return:
        """
        # Adding module transforms
        data.add_transforms(self.get_dataset_transforms())
        # Data loader default options
        data_loader_options.setdefault("shuffle", False)
        data_loader_options.setdefault("drop_last", False)
        data_loader_options.setdefault("pin_memory", True)
        # Building data loader
        loader = DataLoader(data, **data_loader_options)
        # Running generic inference loop
        return generic_inference(self, loader, self.__call__, zip)

    def get_dataset_transforms(self):
        """Returns callable that transform the input data before the forward pass

        :return: List of transforms
        """
        raise NotImplementedError()
