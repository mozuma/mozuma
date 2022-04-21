from collections import OrderedDict

import torch


def state_dict_get(
    state_dict: "OrderedDict[str, torch.Tensor]", key: str
) -> "OrderedDict[str, torch.Tensor]":
    """Allows to get the content of a module given by key in a torch state_dict

    Example:
        Getting the key `a`
        from state dict `{"a.conv": 1, "a.fc": 2, "b.conv": 10}`
        will return `{"conv": 1, "fc": 2}`
    """
    key_with_dot = f"{key}."

    return OrderedDict(
        [
            (k[len(key_with_dot) :], v)
            for k, v in state_dict.items()
            if k.startswith(key_with_dot)
        ]
    )


def state_dict_combine(
    **named_state_dicts: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """Combine state dicts by prefixing keys with the named_state_dicts.keys(

    Example:
        Combining `a={"conv": 1, "fc": 2}` with `b={"conv": 10}`
        returns `{"a.conv": 1, "a.fc": 2, "b.conv": 10}`
    """
    return OrderedDict(
        [
            (f"{state_dict_prefix}.{k}", v)
            for state_dict_prefix, state_dict in named_state_dicts.items()
            for k, v in state_dict.items()
        ]
    )
