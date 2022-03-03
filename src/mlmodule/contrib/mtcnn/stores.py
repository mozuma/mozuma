from collections import OrderedDict
from typing import List, NoReturn

from mlmodule.contrib.mtcnn._mtcnn import MLModuleMTCNN
from mlmodule.contrib.mtcnn.modules import TorchMTCNNModule
from mlmodule.v2.states import StateKey, StateType
from mlmodule.v2.stores.abstract import AbstractStateStore


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

        # Getting a pretrained MTCNN from facenet
        pretrained_mtcnn = MLModuleMTCNN(pretrained=True)
        # Adding the mtcnn prefix to all keys
        pretrained_dict = OrderedDict(
            [
                (f"mtcnn.{key}", value)
                for key, value in pretrained_mtcnn.state_dict().items()
            ]
        )

        # Loading the dictionnary to the model
        model.load_state_dict(pretrained_dict)
