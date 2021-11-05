import dataclasses
from typing import Any, Callable, Generic, TypeVar, cast

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from mlmodule.v2.torch.datasets import TorchDatasetTransformsWrapper
from mlmodule.v2.torch.models import TorchModel
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.results import AbstractResultsProcessor
from mlmodule.v2.torch.runners import TorchInferenceRunner

_Input = TypeVar("_Input")
_Result = TypeVar("_Result")


@dataclasses.dataclass
class DataLoaderFactory:
    transform_func: Callable
    data_loader_options: dict = dataclasses.field(default_factory=dict)

    def __call__(self, dataset) -> DataLoader:
        return DataLoader(
            cast(Dataset, TorchDatasetTransformsWrapper(
                dataset=dataset,
                transform_func=self.transform_func
            )),
            **self.data_loader_options
        )


@dataclasses.dataclass
class BaseInferenceRunnerFactory(Generic[_Input, _Result]):
    model: TorchModel
    results_processor: AbstractResultsProcessor[Any, _Result]
    options: TorchRunnerOptions

    def get_runner(self) -> TorchInferenceRunner[_Input, _Result]:
        return TorchInferenceRunner[_Input, _Result](
            model=self.model,
            data_loader_factory=DataLoaderFactory(
                transform_func=Compose(self.model.get_dataset_transforms()),
                data_loader_options=self.options.data_loader_options
            ),
            results_processor=self.results_processor,
            device=self.options.device,
            tqdm_enabled=self.options.tqdm_enabled
        )
