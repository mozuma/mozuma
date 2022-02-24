import dataclasses
import re
from typing import Optional, Tuple

VALID_STATE_ARCH_NAMES = re.compile(r"^[\w\-\d]+$")
"""The pattern for valid architecture name in
[`StateType.architecture`][mlmodule.v2.states.StateType].
It must be alphanumeric characters separated with dashes (i.e. `[\\w\\-\\d]`)"""


@dataclasses.dataclass(frozen=True)
class StateType:
    """Definition for a type of state

    A state type is used to identify states that can be re-used accross models.

    For instance, the weights from ResNet18 with 1000 classes pretrained on ImageNet
    can be reused to initialised the weights of a ResNet18 for binary classification.
    In this scenario, the state type of the ResNet trained on ImageNet
    can be re-used to partially initialize the binary classification ResNet18.

    This is also used for models like key-frames extraction.
    Key-frames extraction does not define a new weights architecture,
    it is a wrapper around an image encoder.
    Therefore, any state that can be loaded into the image encoder,
    can also be loaded into the key-frame extractor.
    They share the same state type.

    As a convention, two state types are compatible when `backend` and `architecture`
    attributes are the same. This is implemented in the
    [`compatible_with`][mlmodule.v2.states.StateType.compatible_with]
    method.

    Attributes:
        backend (str): The model backend. For instance `pytorch`.
        architecture (str): Identifier for the architecture (i.e. `torchresnet18`...).
            Must match the [`VALID_STATE_ARCH_NAMES`][mlmodule.v2.states.VALID_STATE_ARCH_NAMES] pattern
        extra (Optional[Tuple[str]]): Additional information to identify
            architecture variants (number of output classes...)
    """

    backend: str
    architecture: str
    extra: Optional[Tuple[str]] = None

    def compatible_with(self, other: "StateType") -> bool:
        """Tells whether two architecture are compatible with each other.

        Arguments:
            other (StateType): The other architecture to compare

        Returns:
            bool: `true` if `backend` and `architecture` attributes match.
        """
        return self.backend == other.backend and self.architecture == other.architecture


@dataclasses.dataclass(frozen=True)
class StateKey:
    """Identifier for a state of a trained model

    Attributes:
        state_type (StateType): Identifies the [type of state][mlmodule.v2.states.StateType]
        training_id (str): Identifies the training activity that was used to get to this state
    """

    state_type: StateType
    training_id: str
