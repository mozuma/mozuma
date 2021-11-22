import dataclasses
from typing import Any, BinaryIO, List, Optional

from mlmodule.contrib.keyframes.modules import ResNet18VideoFrameEncoder
from mlmodule.contrib.keyframes.results import KeyFramesSelector
from mlmodule.frames import FrameOutputCollection
from mlmodule.v2.base.models import AbstractModelStore, ProviderModelStore
from mlmodule.v2.torch.datasets import TorchDataset
from mlmodule.v2.torch.factories import AbstractTorchInferenceRunnerFactory
from mlmodule.v2.torch.options import TorchRunnerOptions


@dataclasses.dataclass
class KeyFramesInferenceFactory(
    AbstractTorchInferenceRunnerFactory[
        TorchDataset[Any, BinaryIO], ResNet18VideoFrameEncoder, List[FrameOutputCollection]
    ]
):
    options: TorchRunnerOptions
    model_store: Optional[AbstractModelStore] = None

    def __post_init__(self):
        # Raise an error if the batch size is set to something different from 1
        data_loader_options = self.options.data_loader_options.copy()
        data_loader_options.setdefault('batch_size', 1)
        if data_loader_options['batch_size'] != 1:
            raise ValueError("VideoFramesEncoder doesn't support a batch_size != 1")
        self.options = dataclasses.replace(self.options, data_loader_options=data_loader_options)

    def get_results_processor(self) -> KeyFramesSelector:
        return KeyFramesSelector()

    def get_model(self) -> ResNet18VideoFrameEncoder:
        return ResNet18VideoFrameEncoder()

    def get_model_store(self) -> AbstractModelStore:
        return self.model_store or ProviderModelStore()
