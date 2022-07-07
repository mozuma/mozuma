import dataclasses
import re
from typing import Tuple

VALID_NAMES = re.compile(r"^[\w\-\d]+$")
"""The pattern for valid architecture name in
[`StateType.architecture`][mozuma.states.StateType].
It must be alphanumeric characters separated with dashes (i.e. `[\\w\\-\\d]`)"""


def validate_name(val, field_name):
    if not VALID_NAMES.match(val):
        raise ValueError(
            f"Invalid characters found in state type attribute {field_name}={val}"
        )


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
    [`is_compatible_with`][mozuma.states.StateType.is_compatible_with]
    method.

    Warning:
        All string attributes must match the [`VALID_NAMES`][mozuma.states.VALID_NAMES] pattern.

    Attributes:
        backend (str): The model backend. For instance `pytorch`.
        architecture (str): Identifier for the architecture (e.g. `torchresnet18`...).
        extra (tuple[str, ...]): Additional information to identify
            architecture variants (number of output classes...).
    """

    backend: str
    architecture: str
    extra: Tuple[str, ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        validate_name(self.backend, "backend")
        validate_name(self.architecture, "architecture")
        for e in self.extra:
            validate_name(e, "extra")

    def is_compatible_with(self, other: "StateType") -> bool:
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

    Warning:
        All string attributes must match the [`VALID_NAMES`][mozuma.states.VALID_NAMES] pattern.

    Attributes:
        state_type (StateType): Identifies the [type of state][mozuma.states.StateType]
        training_id (str): Identifies the training activity that was used to get to this state.
    """

    state_type: StateType
    training_id: str

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        validate_name(self.training_id, "training_id")
