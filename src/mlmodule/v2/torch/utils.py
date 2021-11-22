import torch


def resolve_default_torch_device() -> torch.device:
    return (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
