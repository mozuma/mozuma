from collections import OrderedDict
from typing import List, NoReturn

import torch

from mozuma.models.vinvl.modules import TorchVinVLDetectorModule
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

STATE_DICT_URL = "https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth"


class VinVLStore(AbstractStateStore[TorchVinVLDetectorModule]):
    """Pre-trained model states for VinVL

    These are identified by `training_id=vinvl`.
    """

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        if state_type != StateType(backend="pytorch", architecture="vinvl-vg-x152c4"):
            return []
        return [StateKey(state_type=state_type, training_id="vinvl")]

    def save(self, model: TorchVinVLDetectorModule, training_id: str) -> NoReturn:
        raise NotImplementedError("Cannot save to VinVL state store.")

    def load(self, model: TorchVinVLDetectorModule, state_key: StateKey) -> None:
        # Checking model compatibility
        super().load(model, state_key)
        if state_key.training_id != "vinvl":
            raise ValueError(
                f"Cannot find training_id = {state_key.training_id} for VinVL store"
            )

        pretrained_dict: "OrderedDict[str, torch.Tensor]" = (
            torch.hub.load_state_dict_from_url(STATE_DICT_URL)
        )
        cleaned_pretrained_dict = OrderedDict(
            [(k.replace("module.", "vinvl."), v) for k, v in pretrained_dict.items()]
        )
        return model.load_state_dict(cleaned_pretrained_dict)
