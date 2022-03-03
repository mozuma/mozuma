from io import BytesIO
from typing import Mapping, OrderedDict

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


def save_state_dict_to_bytes(obj) -> bytes:
    f = BytesIO()
    torch.save(obj, f)
    f.seek(0)
    return f.read()


def send_batch_to_device(batch, device: torch.device):
    if isinstance(batch, tuple):
        return tuple(send_batch_to_device(b, device) for b in batch)
    elif isinstance(batch, list):
        return [send_batch_to_device(b, device) for b in batch]
    elif hasattr(batch, "to"):
        return batch.to(device)
    else:
        return batch
