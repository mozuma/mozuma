from typing_extensions import Protocol

from mozuma.labels.base import LabelSet
from mozuma.states import StateType


class ModelWithState(Protocol):
    """Protocol of a model with internal state (weights)

    It defines two functions `set_state` and `get_state`.

    Attributes:
        state_type (StateType): Type of the model state,
            see [states](../references/states.md) for more information.
    """

    @property
    def state_type(self) -> StateType:
        """Type of the model state

        See [states](../references/states.md) for more information.
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


class ModelWithLabels:
    """Model that predicts scores for labels

    It defines the `get_labels` function
    """

    def get_labels(self) -> LabelSet:
        """Getter for the model's [`LabelSet`][mozuma.labels.base.LabelSet]

        Returns:
            LabelSet: The label set corresponding to returned label scores
        """
