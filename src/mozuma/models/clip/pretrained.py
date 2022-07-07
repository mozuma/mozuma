from mozuma.models.clip.image import CLIPImageModule
from mozuma.models.clip.text import CLIPTextModule
from mozuma.stores import load_pretrained_model


def torch_clip_image_encoder(*args, **kwargs) -> CLIPImageModule:
    """Pre-trained CLIP image encoder

    Args:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """
    return load_pretrained_model(CLIPImageModule(*args, **kwargs), training_id="clip")


def torch_clip_text_encoder(*args, **kwargs) -> CLIPTextModule:
    """Pre-trained CLIP text encoder

    Args:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """
    return load_pretrained_model(CLIPTextModule(*args, **kwargs), training_id="clip")
