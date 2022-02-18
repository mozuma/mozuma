from typing import Optional, Set

from typing_extensions import Protocol

from mlmodule.labels.base import LabelSet


class ModelWithState(Protocol):
    """Protocol of a model with internal state (weights)

    It defines two functions `set_state` and `get_state`."""

    def state_architecture(self) -> str:
        """Architecture of the returned state of `get_state` function

        Returns:
            str: State architecture ID
        """

    def compatible_state_architectures(self) -> Set[str]:
        """Set of compatible state architectures.

        This is used to identify model weights that are compatible with this model `set_state` method.

        For instance, key-frames extraction works internally with an image encoder.
        In this case, there is no key-frames specific model architecture. Therefore,
        `compatible_state_architectures`
        should return the architecture of the `image_encoder` (most likely `pytorch-{resnet_arch}`).

        This can also be used for transfer learning. In this case,
        it returns all other state architectures
        from which the model can be initialised.

        Returns:
            Set[str]: An identifier of the compatible state architecture
        """

    def set_state(self, state: bytes) -> None:
        """Set the model internal state

        Arguments:
            state (bytes): Serialised state as bytes
        """

    def get_state(self) -> bytes:
        """Get the model internal state

        Returns:
            bytes: Serialised state as bytes
        """


class ModelWithStateFromProvider(ModelWithState):
    """Protocol of a model that can be initialised from a state provider

    The provider is usually the paper's author or a library such as `torchvision`.

    This is used to trace how we created the state files in MLModule repository.
    """

    def provider_state_architectures(self) -> Set[str]:
        """State architectures supported by the provider

        Returns:
            Set[str]: A set of string to identifying the supported states of the provider
        """

    def set_state_from_provider(self, state_arch: Optional[str] = None) -> None:
        """Set the model internal state with original weights

        Arguments:
            state_arch (str): Must be one of `provider_state_architectures`
        """


class ModelWithLabels:
    """Model that predicts scores for labels

    It defines the `get_labels` function
    """

    def get_labels(self) -> LabelSet:
        """Getter for the model's [`LabelSet`][mlmodule.labels.base.LabelSet]

        Returns:
            LabelSet: The label set corresponding to returned label scores
        """
