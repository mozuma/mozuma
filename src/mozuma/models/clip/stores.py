from typing import List, NoReturn

import clip

from mozuma.models.clip.base import BaseCLIPModule
from mozuma.models.clip.utils import CLIP_SAFE_NAME_MAPPING
from mozuma.states import StateKey, StateType
from mozuma.stores.abstract import AbstractStateStore

VALID_CLIP_STATE_TYPE = {
    StateType(backend="pytorch", architecture=f"clip-{model_type}-{model_name}")
    for model_type in ("text", "image")
    for model_name in CLIP_SAFE_NAME_MAPPING
}


class CLIPStore(AbstractStateStore[BaseCLIPModule]):
    """Pre-trained model states by OpenAI CLIP

    These are identified by `training_id=clip`.
    """

    def get_state_keys(self, state_type: StateType) -> List[StateKey]:
        if state_type not in VALID_CLIP_STATE_TYPE:
            return []
        return [StateKey(state_type=state_type, training_id="clip")]

    def save(self, model: BaseCLIPModule, training_id: str) -> NoReturn:
        raise NotImplementedError("Cannot save to CLIP state store.")

    def load(self, model: BaseCLIPModule, state_key: StateKey) -> None:
        # Checking model compatibility
        super().load(model, state_key)

        # Extracting the clip model name
        clip_safe_name = state_key.state_type.architecture.split("-")[-1]
        clip_model_name = CLIP_SAFE_NAME_MAPPING[clip_safe_name]

        # Getting the model state
        clip_pretrained, _ = clip.load(clip_model_name, jit=False)
        pretrained_state = clip_pretrained.state_dict()

        # Loading into CLIP
        model.load_full_clip_state_dict(pretrained_state)
