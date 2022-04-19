import logging
from io import BytesIO
from typing import Any, Dict, Mapping, Optional, OrderedDict, Sequence, Tuple, Union

import torch
from PIL.Image import Image


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


def send_batch_to_device(batch, device: torch.device, non_blocking: bool = False):
    if isinstance(batch, tuple):
        return tuple(send_batch_to_device(b, device, non_blocking) for b in batch)
    elif isinstance(batch, list):
        return [send_batch_to_device(b, device, non_blocking) for b in batch]
    elif hasattr(batch, "to"):
        return batch.to(device, non_blocking=non_blocking)
    else:
        return batch


def apply_mode_to_image(image: Image, mode: str) -> Image:
    if image.mode == mode:
        return image
    return image.convert(mode)


def prepare_batch_for_training(
    batch_wrapper: Tuple,  # type: Tuple[_DatasetType, _TargetsType]
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    batch, target = batch_wrapper

    # Convert targets from lists to tensors
    target_tensor = torch.tensor(target)

    # Sending data on device
    return (
        send_batch_to_device(batch, device=device, non_blocking=non_blocking),
        target_tensor.to(device, non_blocking=non_blocking),
    )


def disable_ignite_logger(logger_name: str, mlmodule_logger: logging.Logger) -> None:
    # Some of Ignite's loggers logs a bunch of infos which we don't want,
    # such as those from auto_dataloader and auto_model.
    # Thus, keep them only if the current's logger level drops below INFO
    if mlmodule_logger.level >= logging.INFO:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_evaluation_metrics(
    logger: logging.Logger,
    epoch: int,
    elapsed: float,
    tag: str,
    metrics: Dict[str, Any],
):
    metrics_output = "\n".join([f"\t{k}: {v:.2f}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )
