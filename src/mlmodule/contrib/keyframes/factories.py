import dataclasses
from typing import List

from mlmodule.contrib.keyframes.modules import VideoFramesEncoder
from mlmodule.contrib.keyframes.results import KeyFramesSelector
from mlmodule.frames import FrameOutputCollection
from mlmodule.v2.torch.datasets import VideoFramesDataset
from mlmodule.v2.torch.factories import BaseInferenceRunnerFactory
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


@dataclasses.dataclass
class KeyFramesInferenceFactory:
    model: VideoFramesEncoder
    options: TorchRunnerOptions

    def __post_init__(self):
        # Raise an error if the batch size is set to something different from 1
        data_loader_options = self.options.data_loader_options.copy()
        data_loader_options.setdefault('batch_size', 1)
        if data_loader_options['batch_size'] != 1:
            raise ValueError("VideoFramesEncoder doesn't support a batch_size != 1")
        self.options = dataclasses.replace(self.options, data_loader_options=data_loader_options)

    def get_runner(self) -> TorchInferenceRunner[VideoFramesDataset, List[FrameOutputCollection]]:
        return BaseInferenceRunnerFactory[VideoFramesDataset, List[FrameOutputCollection]](
            model=self.model,
            results_processor=KeyFramesSelector(),
            options=self.options
        ).get_runner()
