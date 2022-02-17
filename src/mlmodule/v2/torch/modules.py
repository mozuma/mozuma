import abc
from io import BytesIO
from typing import Callable, Generic, List, TypeVar

import torch

from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.torch.utils import save_state_dict_to_bytes

# Type of data of a batch passed to the forward function
_BatchType = TypeVar("_BatchType")
_BatchPredictionArrayType = TypeVar("_BatchPredictionArrayType")


class TorchMlModule(torch.nn.Module, Generic[_BatchType, _BatchPredictionArrayType]):
    """
    Base `torch.nn.Module` for PyTorch models implemented in MLModule.

    A valid subclass of [`TorchMlModule`][mlmodule.v2.torch.modules.TorchMlModule]
    **must** implement the following method:

    - [`forward_predictions`][mlmodule.v2.torch.modules.TorchMlModule.forward_predictions]

    And can optionally implement:

    - [`get_dataset_transforms`][mlmodule.v2.torch.modules.TorchMlModule.get_dataset_transforms]

    Attributes:
        device (torch.device): Mandatory PyTorch device attribute to initialise model.

    Example:
        This would define a simple PyTorch model consisting of fully connected layer.

        ```python
        from mlmodule.v2.torch.modules import TorchMlModule
        from torchvision import transforms


        class FC(TorchMlModule[torch.Tensor]):

            def __init__(self, device: torch.device = torch.device("cpu")):
                super().__init__(device=device)
                self.fc = nn.Linear(512, 512)

            def forward_predictions(
                self, batch: torch.Tensor
            ) -> BatchModelPrediction[torch.Tensor]:
                return BatchModelPrediction(features=self.fc(batch))

            def get_dataset_transforms(self) -> List[Callable]:
                return [transforms.ToTensor()]
        ```

    Note:
        This is a generic class taking a `_BatchType` type argument.
        This corresponds to the type of data the
        [`forward_predictions`][mlmodule.v2.torch.modules.TorchMlModule.forward_predictions]
        will receive. It is most likely `torch.Tensor`
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def forward_predictions(
        self, batch: _BatchType
    ) -> BatchModelPrediction[_BatchPredictionArrayType]:
        """Forward pass of the model

        Applies the module on a batch and returns all potentially interesting data point (features, labels...)

        Arguments:
            batch (_BatchType): the batch of data to process

        Returns:
            BatchModelPrediction[_BatchPredictionArrayType]:
                Prediction object with the keys `features`, `label_scores`...

        Note:
            This method **must** be implemented in subclasses
        """

    def set_state(self, state: bytes) -> None:
        state_dict = torch.load(BytesIO(state), map_location=self.device)
        self.load_state_dict(state_dict)

    def get_state(self) -> bytes:
        return save_state_dict_to_bytes(self.state_dict())

    def get_dataset_transforms(self) -> List[Callable]:
        """Transforms to apply to the input [dataset][mlmodule.v2.torch.datasets.TorchDataset].

        Note:
            By default, this method returns an empty list (meaning no transformation)
            but in most cases, this will need to be overridden.

        Returns:
            List[Callable]: A list of callables that will be used to transform the input data.
        """
        return []
