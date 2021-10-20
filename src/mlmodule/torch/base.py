import abc
import pickle
from typing import Any, Dict, Optional, Union, List, Tuple, Generic, TypeVar

import boto3
import numpy as np
import torch
import torch.nn as nn
from io import BytesIO
from torch.utils.data.dataloader import DataLoader
from typing_extensions import Protocol

from mlmodule.base import BaseMLModule, LoadDumpMixin
from mlmodule.box import BBoxCollection, BBoxOutputArrayFormat
from mlmodule.torch.handlers import results_handler_bbox, results_handler_numpy_array
from mlmodule.torch.utils import (
    generic_inference,
    torch_apply_state_to_partial_model
)
from mlmodule.types import StateDict


_IndexType = TypeVar('_IndexType', covariant=True)
_InputDataType = TypeVar('_InputDataType', covariant=True)
_ForwardRetType = TypeVar('_ForwardRetType')
_InferenceRetType = TypeVar('_InferenceRetType')


class MLModuleDatasetProtocol(Protocol[_IndexType, _InputDataType]):
    """Pytorch dataset protocol with __len__"""

    def __getitem__(self, index: int) -> Tuple[_IndexType, _InputDataType]: ...

    def __len__(self) -> int: ...


class AbstractTorchMLModule(
        BaseMLModule,
        nn.Module,
        LoadDumpMixin,
        Generic[_IndexType, _InputDataType, _ForwardRetType, _InferenceRetType]
):

    state_dict_key: Optional[str] = None
    default_batch_size = 128

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or self._resolve_device()

    def _torch_load(self, f, map_location=None, pickle_module=pickle, **pickle_load_args) -> Any:
        """Safe method to load the state dict directly on the right device"""
        map_location = map_location or self.device
        return torch.load(f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)

    @classmethod
    def _resolve_device(cls):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def load(self, fp=None, pretrained_getter_opts: Dict[str, Any] = None):
        """Loads model from file or from a default pretrained model if `fp=None`

        :param pretrained_getter_opts: Passed to get_default_pretrained_dict
        :param fp:
        :return:
        """
        with self.metrics.measure('time_load_weigths'):
            # Getting state dict
            if fp:
                fp.seek(0)
                state = self._torch_load(fp)
            else:
                # Getting default pretrained state dict
                state = self.get_default_pretrained_state_dict(**(pretrained_getter_opts or {}))

            # Loading state
            self.load_state_dict(state)
            self.to(self.device)

        return self

    def dump(self, fp):
        with self.metrics.measure('time_dump_weights'):
            torch.save(self.state_dict(), fp)

    def get_default_pretrained_state_dict(
            self,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None
    ) -> StateDict:
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
        pretrained_state_dict = self._torch_load(f, map_location=lambda storage, loc: storage)
        return torch_apply_state_to_partial_model(self, pretrained_state_dict)

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
        data_loader_options["batch_size"] = data_loader_options.get("batch_size") or self.default_batch_size
        # We send to pin memory only if using CUDA device
        data_loader_options.setdefault(
            "pin_memory", self.device != torch.device('cpu'))
        # Building data loader
        return DataLoader(data, **data_loader_options)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> _ForwardRetType:
        """Forward pass of the module"""

    @abc.abstractmethod
    def results_handler(
            self,
            acc_results: Tuple[List[_IndexType], _InferenceRetType],
            new_indices: Union[torch.Tensor, List],
            new_output: _ForwardRetType
    ) -> Tuple[List[_IndexType], _InferenceRetType]:
        """Runs at the end of the inference loop

        :param acc_results: The results accumulated
        :param new_indices: The new indices of data for the current batch
        :param new_output: The new data for the current batch
        :return: new accumulated results
        """

    def inference(self, *args, **kwargs) -> _ForwardRetType:
        return self.forward(*args, **kwargs)

    def bulk_inference(
            self,
            data: MLModuleDatasetProtocol[_IndexType, _InputDataType],
            **options
    ) -> Optional[Tuple[List[_IndexType], _InferenceRetType]]:    # Returns None is dataset length = 0
        """Run the model against all elements in data"""
        self.metrics.add('dataset_size', len(data))
        loader = self.get_data_loader(data, **(options.get('data_loader_options', {})))

        with self.metrics.measure('time_generic_inference'):
            # Running inference batch loop
            return generic_inference(
                self, loader, self.inference, self.results_handler, self.device,
                result_handler_options=options.get('result_handler_options'),
                inference_options=options.get('inference_options'),
                tqdm_enabled=options.get('tqdm_enabled', False)
            )

    def get_dataset_transforms(self):
        """Returns callable that transform the input data before the forward pass

        :return: List of transforms
        """
        return []


class TorchMLModuleFeatures(
        AbstractTorchMLModule[
            _IndexType,             # Type of the data index (generic)
            _InputDataType,         # Type of the input dataset
            torch.Tensor,           # Type of the data returned by forward
            np.ndarray              # Type of data returned by bulk_inference
        ]
):

    def results_handler(
            self,
            acc_results: Tuple[List[_IndexType], np.ndarray],
            new_indices: Union[torch.Tensor, List],
            new_output: torch.Tensor
    ) -> Tuple[List[_IndexType], np.ndarray]:
        return results_handler_numpy_array(acc_results, new_indices, new_output)


# for compatibility reasons
BaseTorchMLModule = TorchMLModuleFeatures


class TorchMLModuleBBox(
        AbstractTorchMLModule[
            _IndexType,
            _InputDataType,
            BBoxOutputArrayFormat,
            List[BBoxCollection]
        ]
):

    def results_handler(
            self,
            acc_results: Tuple[List[_IndexType], List[BBoxCollection]],
            new_indices: Union[torch.Tensor, List],
            new_output: BBoxOutputArrayFormat
    ) -> Tuple[List[_IndexType], List[BBoxCollection]]:
        return results_handler_bbox(acc_results, new_indices, new_output)
