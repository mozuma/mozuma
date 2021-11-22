from typing import Any, BinaryIO, Dict, List, Optional, OrderedDict, Tuple, TypeVar, Union

import torch
from torchvision.transforms import Compose

from mlmodule.contrib.keyframes.modules import ResNet18VideoFrameEncoder
from mlmodule.contrib.keyframes.results import KeyFramesSelector
from mlmodule.frames import FrameOutputCollection
from mlmodule.torch.base import AbstractTorchMLModule
from mlmodule.types import StateDict
from mlmodule.v2.base.models import ProviderModelStore
from mlmodule.v2.torch.datasets import TorchDataset
from mlmodule.v2.torch.factories import DataLoaderFactory
from mlmodule.v2.torch.runners import TorchInferenceRunner


_IndexType = TypeVar("_IndexType", covariant=True)


class TorchMLModuleKeyFrames(
    AbstractTorchMLModule[
        _IndexType, TorchDataset[Any, BinaryIO], torch.Tensor, List[FrameOutputCollection]
    ]
):
    def __init__(self, device: torch.device = None):
        super().__init__(device=device)
        self.keyframes = ResNet18VideoFrameEncoder()
        self.keyframes.to(self.device)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        self.keyframes.load_state_dict(state_dict, strict=strict)

    def state_dict(self):
        return self.keyframes.state_dict()

    def load(self, fp=None, pretrained_getter_opts: Dict[str, Any] = None):
        """Loads model from file or from a default pretrained model if `fp=None`

        :param pretrained_getter_opts: Passed to get_default_pretrained_dict
        :param fp:
        :return:
        """
        # Getting state dict
        if fp:
            fp.seek(0)
            state = self._torch_load(fp)
        else:
            # Getting default pretrained state dict
            state = self.get_default_pretrained_state_dict(
                **(pretrained_getter_opts or {})
            )

        # Loading state
        self.keyframes.load_state_dict(state)

        return self

    def dump(self, fp):
        with self.metrics.measure("time_dump_weights"):
            torch.save(self.keyframes.state_dict(), fp)

    def get_default_pretrained_state_dict(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> StateDict:
        ProviderModelStore().load(self.keyframes, device=self.device)
        return self.keyframes.state_dict()

    def get_default_pretrained_state_dict_from_provider(self) -> StateDict:
        return self.get_default_pretrained_state_dict()

    def bulk_inference(
        self, data: TorchDataset[Any, BinaryIO], **options
    ) -> Optional[
        Tuple[List[_IndexType], List[FrameOutputCollection]]
    ]:  # Returns None is dataset length = 0
        """Run the model against all elements in data"""
        self.metrics.add("dataset_size", len(data))
        data_loader_options = options.get("data_loader_options", {})

        return TorchInferenceRunner(
            model=self.keyframes,
            data_loader_factory=DataLoaderFactory(
                transform_func=Compose(self.keyframes.get_dataset_transforms()),
                data_loader_options=data_loader_options,
            ),
            results_processor=KeyFramesSelector(),
            device=self.device,
            tqdm_enabled=options.get("tqdm_enabled", False),
        ).bulk_inference(data)

    def results_handler(
        self,
        acc_results: Tuple[List[_IndexType], torch.Tensor],
        new_indices: Union[torch.Tensor, List],
        new_output: torch.Tensor,
    ) -> Tuple[List[_IndexType], torch.Tensor]:
        pass
