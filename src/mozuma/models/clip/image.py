from collections import OrderedDict
from typing import Callable, List

import torch

from mozuma.models.clip.base import BaseCLIPModule
from mozuma.models.clip.transforms import get_image_transform
from mozuma.models.clip.utils import get_clip_module


class CLIPImageModule(BaseCLIPModule):
    """Image encoder of the CLIP model

    Attributes:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """

    def __init__(
        self, clip_model_name: str, device: torch.device = torch.device("cpu")
    ):
        super().__init__(clip_model_name, "image", device=device)

        # Loading CLIP module
        clip_module = get_clip_module(self.clip_model_name)

        # Populating image encoder attributes
        self.visual = clip_module.visual

        self.convert_weights()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of image encoder

        Arguments:
            batch (torch.Tensor): Batch of images

        Returns:
            torch.Tensor: The image features
        """
        return self.visual(batch.type(self._dtype))

    def get_dataset_transforms(self) -> List[Callable]:
        """Dataset transform to resize and preprocess images"""
        return [get_image_transform(self.visual.input_resolution)]

    def load_full_clip_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]"):
        # Filtering the visual modules keys
        visual_state = OrderedDict(
            [
                (key, value)
                for key, value in state_dict.items()
                if key.startswith("visual")
            ]
        )

        # Loading the state weights
        self.load_state_dict(visual_state)
