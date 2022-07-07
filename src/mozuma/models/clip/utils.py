from typing import Dict

import clip
import torch
from clip.model import CLIP

from mozuma.models.clip.parameters import PARAMETERS


def get_clip_module(clip_model_name: str) -> CLIP:
    """Loads the CLIP from a model name

    Arguments:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
    """
    return CLIP(*PARAMETERS[clip_model_name].values())


def clip_tokenize_single(text: str) -> torch.LongTensor:
    """Takes a str and returns its tokenized version"""
    return clip.tokenize(text)[0]


def sanitize_clip_model_name(clip_model_name: str) -> str:
    """Removes dash and slash characters"""
    return clip_model_name.lower().replace("-", "").replace("/", "")


CLIP_SAFE_NAME_MAPPING: Dict[str, str] = {
    sanitize_clip_model_name(p): p for p in PARAMETERS
}
