from collections import OrderedDict
from typing import List, NoReturn

import torch

from mozuma.helpers.gdrive import download_file_from_google_drive
from mozuma.models.magface.modules import TorchMagFaceModule
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore


class MagFaceStore(AbstractStateStore[TorchMagFaceModule]):
    """Pre-trained model states by MagFace (<https://github.com/IrvingMeng/MagFace>)

    These are identified by `training_id=magface`.
    """

    GOOGLE_DRIVE_FILE_ID = "1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H"

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        if state_type != StateType(backend="pytorch", architecture="magface"):
            return []
        return [StateKey(state_type=state_type, training_id="magface")]

    def save(self, model: TorchMagFaceModule, training_id: str) -> NoReturn:
        raise NotImplementedError("Cannot save to MagFace state store.")

    def load(self, model: TorchMagFaceModule, state_key: StateKey) -> None:
        # Checking model compatibility
        super().load(model, state_key)
        if state_key.training_id != "magface":
            raise ValueError(
                f"Cannot find training_id = {state_key.training_id} for MagFace store"
            )

        # Downloading state dict from Google Drive
        pretrained_state_dict: "OrderedDict[str, torch.Tensor]" = torch.load(
            download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID),
            map_location=model.device,
        )["state_dict"]
        cleaned_state_dict = OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.startswith("features.module."):
                new_k = ".".join(k.split(".")[2:])
                cleaned_state_dict[new_k] = v

        # Removing deleted layers from state dict and updating the other with pretrained data
        return model.load_state_dict(cleaned_state_dict)
