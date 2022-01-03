import logging
from typing import Callable, Dict, List, Union, cast

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def torch_apply_state_to_partial_model(
    partial_model: torch.nn.Module, pretrained_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Creates a new state dict by updating the partial_model state with matching parameters of pretrained_state_dict

    See https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    :param partial_model:
    :param pretrained_state_dict:
    :return:
    """
    model_dict = partial_model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)

    return model_dict


def generic_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    inference_func: Callable,
    result_handler: Callable,
    device: torch.device,
    result_handler_options=None,
    inference_options=None,
    tqdm_enabled=False,
):
    # Setting model in eval mode
    model.eval()

    # Sending model on device
    model.to(device)

    # Disabling gradient computation
    acc_results = None
    with torch.no_grad():
        # Looping through batches
        # Assume dataset is composed of tuples (item index, batch)
        n_batches = len(data_loader)
        if tqdm_enabled:
            data_loader = tqdm(data_loader)
        for batch_n, (indices, batch, *batch_params) in enumerate(data_loader):
            logger.debug(f"Sending batch number: {batch_n}/{n_batches}")
            # Sending data on device
            batch = batch.to(device)
            acc_results = result_handler(
                acc_results,
                indices,
                inference_func(batch, *batch_params, **(inference_options or {})),
                **(result_handler_options or {}),
            )
            logger.debug(f"Collecting results: {batch_n}/{n_batches}")

    # Returning accumulated results
    return acc_results


def tensor_to_python_list_safe(tensor_or_list: Union[torch.Tensor, List]) -> List:
    """Transforms a tensor into a Python list.

    If the argument is a list, returns is without raising.

    :param tensor_or_list:
    :return:
    """
    ret: list
    if hasattr(tensor_or_list, "tolist"):  # This is a tensor
        ret = cast(torch.Tensor, tensor_or_list).tolist()
    else:
        ret = cast(list, tensor_or_list)
    return ret


def tensor_to_ndarray(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Transforms a tensor to ndarray"""
    if hasattr(arr, "cpu"):
        return cast(torch.Tensor, arr).cpu().numpy()
    else:
        return cast(np.ndarray, arr)


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)
    return output
