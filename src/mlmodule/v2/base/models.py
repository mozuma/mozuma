from typing_extensions import Protocol, runtime_checkable

from mlmodule.labels.base import LabelSet


@runtime_checkable
class ModelWithState(Protocol):
    """Protocol of a model with internal state (weights)

    It defines two functions `set_state` and `get_state`."""

    # Unique identifier for the model
    @property
    def mlmodule_model_uri(self) -> str:
        ...

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
    """Protocol of a model that can be initialised with the original weights

    The provider is usually the paper's author or a library such as `torchvision`."""

    def set_state_from_provider(self) -> None:
        """Set the model internal state with original weights"""
        ...


class ModelWithLabels:
    """Model that predicts scores for labels

    It defines the `get_labels` function
    """

    def get_labels(self) -> LabelSet:
        """Getter for the model's [`LabelSet`][mlmodule.labels.base.LabelSet]

        Returns:
            LabelSet: The label set corresponding to returned label scores
        """
