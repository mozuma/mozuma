from mozuma.models.magface.modules import TorchMagFaceModule
from mozuma.stores import load_pretrained_model


def torch_magface(*args, **kwargs) -> TorchMagFaceModule:
    """Pre-trained MagFace module

    Args:
        device (torch.device): Torch device to initialise the model weights
        remove_bad_faces (bool): Whether to remove the faces with bad quality from the output.
            This will replace features of bad faces with `float("nan")`. Defaults to `False`.
        magnitude_threshold (float): Threshold to remove bad quality faces.
            The higher the stricter. Defaults to `22.5`.
    """
    return load_pretrained_model(
        TorchMagFaceModule(*args, **kwargs), training_id="magface"
    )
