from mozuma.models.vinvl.modules import TorchVinVLDetectorModule
from mozuma.stores import load_pretrained_model


def torch_vinvl_detector(*args, **kwargs) -> TorchVinVLDetectorModule:
    """[VinVL](https://github.com/pzzhang/VinVL) object detection model

    Args:
        score_threshold (float):
        attr_score_threshold (float):
        device (torch.device): PyTorch device attribute to initialise model.
    """
    return load_pretrained_model(
        TorchVinVLDetectorModule(*args, **kwargs), training_id="vinvl"
    )
