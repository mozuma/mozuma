import abc
from io import BytesIO
from typing import Any, Callable, Generic, List, Optional, TypeVar

import torch

from mlmodule.v2.base.predictions import BatchModelPrediction
from mlmodule.v2.states import StateType
from mlmodule.v2.torch.utils import save_state_dict_to_bytes

# Type of data of a batch passed to the forward function
_BatchType = TypeVar("_BatchType")
_ForwardOutputType = TypeVar("_ForwardOutputType")


class TorchMlModule(torch.nn.Module, Generic[_BatchType, _ForwardOutputType]):
    """
    Base `torch.nn.Module` for PyTorch models implemented in MLModule.

    A valid subclass of [`TorchMlModule`][mlmodule.v2.torch.modules.TorchMlModule]
    **must** implement the following method:

    - [`forward`][mlmodule.v2.torch.modules.TorchMlModule.forward]
    - [`to_predictions`][mlmodule.v2.torch.modules.TorchMlModule.to_predictions]
    - [`state_type`][mlmodule.v2.torch.modules.TorchMlModule.state_type]

    And can optionally implement:

    - [`get_dataset_transforms`][mlmodule.v2.torch.modules.TorchMlModule.get_dataset_transforms]
    - [`get_dataloader_collate_fn`][mlmodule.v2.torch.modules.TorchMlModule.get_dataloader_collate_fn]

    Attributes:
        device (torch.device): Mandatory PyTorch device attribute to initialise model.

    Example:
        This would define a simple PyTorch model consisting of fully connected layer.

        ```python
        from mlmodule.v2.states import StateType
        from mlmodule.v2.torch.modules import TorchMlModule
        from torchvision import transforms


        class FC(TorchMlModule[torch.Tensor, torch.Tensor]):

            def __init__(self, device: torch.device = torch.device("cpu")):
                super().__init__(device=device)
                self.fc = nn.Linear(512, 512)

            def forward(
                self, batch: torch.Tensor
            ) -> torch.Tensor:
                return self.fc(batch)

            def to_predictions(
                self, forward_output: torch.Tensor
            ) -> BatchModelPrediction[torch.Tensor]:
                return BatchModelPrediction(features=forward_output)

            @property
            def state_type(self) -> StateType:
                return StateType(
                    backend="pytorch",
                    architecture="fc512x512",
                )

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

    @staticmethod
    def _extract_device_from_args(*args, **kwargs) -> Optional[torch.device]:
        """Extracts the device argument from the `to` method arguments"""
        if "device" in kwargs:
            return kwargs["device"]
        # Otherwise find the first argument matching the expected type of a device torch.device | int
        return next(
            (
                torch.device(a)
                for a in args
                if isinstance(a, int) or isinstance(a, torch.device)
            ),
            None,
        )

    def to(self, *args, **kwargs):
        device = self._extract_device_from_args(*args, **kwargs)
        if device:
            self.device = device
        return super().to(*args, **kwargs)

    @abc.abstractproperty
    def state_type(self) -> StateType:
        """Identifier for the current's model state architecture

        Note:
            PyTorch's model architecture should have the `pytorch` backend

        Returns:
            StateType: State architecture object

        Note:
            This method **must** be implemented in subclasses
        """
        raise NotImplementedError("State architecture should be overridden")

    @abc.abstractmethod
    def forward(self, batch: _BatchType) -> _ForwardOutputType:
        """Forward pass of the model

        Applies the module on a batch and returns all potentially interesting data point (features, labels...)

        Arguments:
            batch (_BatchType): the batch of data to process

        Returns:
            _ForwardOutputType: A tensor or a sequence of tensor with relevant information
                (features, labels, bounding boxes...)

        Note:
            This method **must** be implemented in subclasses
        """

    @abc.abstractmethod
    def to_predictions(
        self, forward_output: _ForwardOutputType
    ) -> BatchModelPrediction[torch.Tensor]:
        """Modifies the output of the forward pass to create the standard BatchModelPrediction object

        Arguments:
            forward_output (_ForwardOutputType): the batch of data to process

        Returns:
            BatchModelPrediction[torch.Tensor]:
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

    def get_dataloader_collate_fn(self) -> Optional[Callable[[Any], Any]]:
        """Optionally returns a collate function to be passed to the data loader

        Note:
            This collate function will be wrapped in `mlmodule.v2.torch.collate.TorchMlModuleCollateFn`.
            This means that the first argument `batch` will not contain
            the indices of the dataset but only the data element.

        Returns:
            Callable[[Any], Any] | None: The collate function to be passed to `TorchMlModuleCollateFn`.
        """
        return None
