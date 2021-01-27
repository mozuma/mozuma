import logging
import torch


logger = logging.getLogger(__name__)


def torch_apply_state_to_partial_model(partial_model, pretrained_state_dict):
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


def generic_inference(model, data_loader, forward_func, result_handler, device):
    # Setting model in eval mode
    model.eval()

    # Sending model on device
    model = model.to(device)

    # Disabling gradient computation
    results = []
    with torch.no_grad():
        # Looping through batches
        # Assume dataset is composed of tuples (item index, batch)
        for batch_n, (indices, batch) in enumerate(data_loader):
            logger.debug(f"Sending batch number: {batch_n}")
            # Sending data on device
            batch = batch.to(device)
            results += result_handler(indices, forward_func(batch).detach())

    # Sorting and returning predictions
    return [x for x in sorted(results, key=lambda i: i[0])]
