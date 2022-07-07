from mozuma.models.mtcnn.modules import TorchMTCNNModule
from mozuma.stores import load_pretrained_model


def torch_mtcnn(*args, **kwargs) -> TorchMTCNNModule:
    """Pre-trained PyTorch's MTCNN face detection module by FaceNet

    Args:
        thresholds (Tuple[float, float, float]): MTCNN threshold hyperparameters
        image_size (Tuple[int, int]): Image size after pre-preprocessing
        min_face_size (int): Minimum face size in pixels
        device (torch.device): Torch device to initialise the model weights

    Returns:
        TorchMTCNNModule: Pre-trained MTCNN model
    """
    return load_pretrained_model(
        TorchMTCNNModule(*args, **kwargs), training_id="facenet"
    )
