from mozuma.models.arcface.modules import TorchArcFaceModule
from mozuma.stores import load_pretrained_model


def torch_arcface_insightface(*args, **kwargs) -> TorchArcFaceModule:
    """ArcFace model pre-trained by InsightFace

    Args:
        device (torch.device): Torch device to initialise the model weights
        remove_bad_faces (bool): Whether to remove the faces with bad quality from the output.
            This will replace features of bad faces with `float("nan")`. Defaults to `False`.
        bad_faces_threshold (float): The cosine similarity distance to reference faces for which we
            consider the face is of bad quality.

    Returns:
        TorchArcFaceModule: Pre-trained ArcFace
    """
    return load_pretrained_model(
        TorchArcFaceModule(*args, **kwargs), training_id="insightface"
    )
