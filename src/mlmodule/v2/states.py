import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class StateArchitecture:
    """Definition for a type of state

    A state architecture is used to identify states that can be re-used accross models.

    For instance, the weights from ResNet18 with 1000 classes pretrained on ImageNet
    can be reused to initialised the weights of a ResNet18 for binary classification.
    In this scenario, the state architecture of the saved weights for the ImageNet ResNet
    is compatible with the binary classification ResNet18.

    This is also used for models like key-frames extraction.
    Key-frames extraction does not define a new architecture,
    it is a wrapper around an image encoder.
    Therefore, any state that can be loaded into the image encoder,
    can also be loaded into the key-frame extractor.
    They share the same state architecture.

    As a convention, two state architectures are compatible when `backend` and `architecture`
    attributes are the same. This is implemented in the
    [`compatible_with`][mlmodule.v2.states.StateArchitecture.compatible_with]
    method.

    Attributes:
        backend (str): The model backend. For instance `pytorch`.
        architecture (str): Identifier for the architecture (i.e. `torchresnet18`...)
        extra (Optional[Tuple[str]]): Additional information to identify
            architecture variants (number of output classes...)
    """

    backend: str
    architecture: str
    extra: Optional[Tuple[str]] = None

    def compatible_with(self, other: "StateArchitecture") -> bool:
        """Tells whether two architecture are compatible with each other.

        Arguments:
            other (StateArchitecture): The other architecture to compare

        Returns:
            bool: `true` if `backend` and `architecture` attributes match.
        """
        return self.backend == other.backend and self.architecture == other.architecture


@dataclasses.dataclass(frozen=True)
class StateIdentifier:
    """Identifier for a state of a trained model

    Attributes:
        state_architecture (StateArchitecture): Identifies the [type of state][mlmodule.v2.states.StateArchitecture]
        training_id (str): Identifies the training activity that was used to get to this state
    """

    state_architecture: StateArchitecture
    training_id: str
