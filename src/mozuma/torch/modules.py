import abc
from io import BytesIO
from typing import Any, Callable, Generic, List, Optional, TypeVar

import torch

from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.utils import save_state_dict_to_bytes

# Type of data of a batch passed to the forward function
_BatchType = TypeVar("_BatchType")
_ForwardOutputType = TypeVar("_ForwardOutputType")


class TorchModel(torch.nn.Module, Generic[_BatchType, _ForwardOutputType]):
    """
    Base `torch.nn.Module` for PyTorch models.

    A valid subclass of [`TorchModel`][mozuma.torch.modules.TorchModel]
    **must** implement the following method:

    - [`forward`][mozuma.torch.modules.TorchModel.forward]
    - [`to_predictions`][mozuma.torch.modules.TorchModel.to_predictions]
    - [`state_type`][mozuma.torch.modules.TorchModel.state_type]

    And can optionally implement:

    - [`get_dataset_transforms`][mozuma.torch.modules.TorchModel.get_dataset_transforms]
    - [`get_dataloader_collate_fn`][mozuma.torch.modules.TorchModel.get_dataloader_collate_fn]

    Attributes:
        device (torch.device): Mandatory PyTorch device attribute to initialise model.
        is_trainable (bool): Flag which indicates if the model is trainable. Default, True.

    Example:
        This would define a simple PyTorch model consisting of fully connected layer.

        ```python
        from mozuma.states import StateType
        from mozuma.torch.modules import TorchModel
        from torchvision import transforms


        class FC(TorchModel[torch.Tensor, torch.Tensor]):

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
        This is a generic class taking a `_BatchType` and `_ForwardOutputType` type argument.
        This corresponds respectively to the type of data the
        [`forward`][mozuma.torch.modules.TorchModel.forward]
        will take as argument and return. It is most likely `torch.Tensor`

    Note:
        By default, MoZuMa models are trainable. Set the `is_trainable`
        parameter to `False` when creating a subclass if it shouldn't be trained.
    """  # noqa: E501

    def __init__(
        self, device: torch.device = torch.device("cpu"), is_trainable: bool = True
    ):
        super().__init__()
        self.device = device
        self.is_trainable = is_trainable

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

        Important:
            This property **must** be implemented in subclasses

        Note:
            PyTorch's model architecture should have the `pytorch` backend

        Returns:
            StateType: State architecture object
        """
        raise NotImplementedError("State architecture should be overridden")

    @abc.abstractmethod
    def forward(self, batch: _BatchType) -> _ForwardOutputType:
        """Forward pass of the model

        Important:
            This method **must** be implemented in subclasses

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

        Important:
            This method **must** be implemented in subclasses

        Arguments:
            forward_output (_ForwardOutputType): the batch of data to process

        Returns:
            BatchModelPrediction[torch.Tensor]:
                Prediction object with the keys `features`, `label_scores`...
        """

    def set_state(self, state: bytes) -> None:
        state_dict = torch.load(BytesIO(state), map_location=self.device)
        self.load_state_dict(state_dict)

    def get_state(self) -> bytes:
        return save_state_dict_to_bytes(self.state_dict())

    def get_dataset_transforms(self) -> List[Callable]:
        """Transforms to apply to the input [dataset][mozuma.torch.datasets.TorchDataset].

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
            This collate function will be wrapped in `mozuma.torch.collate.TorchModelCollateFn`.
            This means that the first argument `batch` will not contain
            the indices of the dataset but only the data element.

        Returns:
            Callable[[Any], Any] | None: The collate function to be passed to `TorchModelCollateFn`.
        """
        return None
