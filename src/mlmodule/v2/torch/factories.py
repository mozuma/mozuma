import abc
import dataclasses
from typing import Any, Callable, Generic, TypeVar, cast

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from mlmodule.v2.base.factories import AbstractRunnerFactory
from mlmodule.v2.base.models import ModelWithState
from mlmodule.v2.torch.datasets import TorchDatasetTransformsWrapper
from mlmodule.v2.torch.models import TorchModel
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.results import AbstractResultsProcessor
from mlmodule.v2.torch.runners import TorchInferenceRunner

_Input = TypeVar("_Input")
_Model = TypeVar("_Model", bound=TorchModel)
_Result = TypeVar("_Result")


@dataclasses.dataclass
class DataLoaderFactory:
    transform_func: Callable
    data_loader_options: dict = dataclasses.field(default_factory=dict)

    def __call__(self, dataset) -> DataLoader:
        return DataLoader(
            cast(
                Dataset,
                TorchDatasetTransformsWrapper(
                    dataset=dataset, transform_func=self.transform_func
                ),
            ),
            **self.data_loader_options
        )


class AbstractTorchInferenceRunnerFactory(
    AbstractRunnerFactory[_Model, TorchInferenceRunner[_Input, _Result]],
    Generic[_Input, _Model, _Result],
):
    options: TorchRunnerOptions

    @abc.abstractmethod
    def get_results_processor(self) -> AbstractResultsProcessor[Any, _Result]:
        """Results processor to be passed to the Torch Inference runner"""

    def get_data_loader_factory(self, model: _Model) -> DataLoaderFactory:
        """"""
        return DataLoaderFactory(
            transform_func=Compose(model.get_dataset_transforms()),
            data_loader_options=self.options.data_loader_options,
        )

    def get_runner(self) -> TorchInferenceRunner[_Input, _Result]:
        model = self.get_model()
        if isinstance(model, ModelWithState):
            # Loading model state
            self.get_model_store().load(model, device=self.options.device)

        return TorchInferenceRunner[_Input, _Result](
            model=model,
            data_loader_factory=self.get_data_loader_factory(model),
            results_processor=self.get_results_processor(),
            device=self.options.device,
            tqdm_enabled=self.options.tqdm_enabled,
        )
