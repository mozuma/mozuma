from collections import OrderedDict
from typing import Mapping

import torch


def resolve_default_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def add_prefix_to_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> "OrderedDict[str, torch.Tensor]":
    ret = OrderedDict()
    for key, value in state_dict.items():
        ret[f"{prefix}.{key}"] = value
    return ret
