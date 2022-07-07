from collections import OrderedDict
from typing import List, NoReturn

from mozuma.models.mtcnn._mtcnn import MoZuMaMTCNN
from mozuma.models.mtcnn.modules import TorchMTCNNModule
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore


class FaceNetMTCNNStore(AbstractStateStore[TorchMTCNNModule]):
    """Pre-trained model states by Facenet for MTCNN

    These are identified by `training_id=facenet`.
    """

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        if state_type != StateType(backend="pytorch", architecture="facenet-mtcnn"):
            return []
        return [StateKey(state_type=state_type, training_id="facenet")]

    def save(self, model: TorchMTCNNModule, training_id: str) -> NoReturn:
        raise NotImplementedError("Cannot save to FaceNet MTCNN state store.")

    def load(self, model: TorchMTCNNModule, state_key: StateKey) -> None:
        # Checking model compatibility
        super().load(model, state_key)
        if state_key.training_id != "facenet":
            raise ValueError(
                f"Cannot find training_id = {state_key.training_id} for FaceNet MTCNN store"
            )

        # Getting a pretrained MTCNN from facenet
        pretrained_mtcnn = MoZuMaMTCNN(pretrained=True)
        # Adding the mtcnn prefix to all keys
        pretrained_dict = OrderedDict(
            [
                (f"mtcnn.{key}", value)
                for key, value in pretrained_mtcnn.state_dict().items()
            ]
        )

        # Loading the dictionnary to the model
        model.load_state_dict(pretrained_dict)
