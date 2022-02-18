import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class StateIdentifier:
    """Identifier for a state of a trained model

    Attributes:
        state_architecture (str): Identifies the type of state
        training_id (str): Identifies the training activity that was used to get to this state
    """

    state_architecture: str
    training_id: Optional[str] = None
