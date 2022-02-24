from typing import Optional, Set

from typing_extensions import Protocol

from mlmodule.labels.base import LabelSet
from mlmodule.v2.states import StateKey, StateType


class ModelWithState(Protocol):
    """Protocol of a model with internal state (weights)

    It defines two functions `set_state` and `get_state`.

    Attributes:
        state_type (StateType): Type of the model state,
            see [states](../references/states.md) for more information.
    """

    state_type: StateType

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

    def provider_state_architectures(self) -> Set[StateType]:
        """State architectures supported by the provider

        Returns:
            Set[StateType]: A set of `StateType` to identifying the supported states of the provider
        """

    def set_state_from_provider(
        self, state_arch: Optional[StateType] = None
    ) -> StateKey:
        """Set the model internal state with original weights

        Arguments:
            state_arch (StateType): Must be one of `provider_state_architectures`

        Returns:
            StateKey:
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
