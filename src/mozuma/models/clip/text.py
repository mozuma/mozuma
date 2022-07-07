from collections import OrderedDict
from typing import Callable, List, cast

import torch

from mozuma.models.clip.base import BaseCLIPModule
from mozuma.models.clip.utils import clip_tokenize_single, get_clip_module


class CLIPTextModule(BaseCLIPModule):
    """Text encoder of the CLIP model

    Attributes:
        clip_model_name (str): Name of the model to load
            (see [CLIP doc](https://github.com/openai/CLIP#clipavailable_models))
        device (torch.device, optional): The PyTorch device to initialise the model weights.
            Defaults to `torch.device("cpu")`.
    """

    def __init__(
        self, clip_model_name: str, device: torch.device = torch.device("cpu")
    ):
        super().__init__(clip_model_name, model_type="text", device=device)

        clip_module = get_clip_module(self.clip_model_name)

        # Populating with text encoder attributes
        self.context_length = clip_module.context_length
        self.vocab_size = clip_module.vocab_size
        self.token_embedding = clip_module.token_embedding
        self.positional_embedding = clip_module.positional_embedding
        self.transformer = clip_module.transformer
        self.ln_final = clip_module.ln_final
        self.text_projection = clip_module.text_projection

        self.convert_weights()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the text encoder

        Arguments:
            batch (torch.Tensor): Batch of texts

        Returns:
            torch.Tensor: The features of the text encoder
        """
        x = cast(torch.Tensor, self.token_embedding(batch)).type(
            self._dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self._dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = cast(torch.Tensor, self.transformer(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = cast(torch.Tensor, self.ln_final(x)).type(self._dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), batch.argmax(dim=-1)] @ self.text_projection

        return x

    def get_dataset_transforms(self) -> List[Callable]:
        """Dataset transforms (tokenizer)"""
        return [clip_tokenize_single]

    def load_full_clip_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]"):
        # Filtering the text modules keys
        text_module_fields = (
            "context_length",
            "vocab_size",
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
        )
        text_state = OrderedDict(
            [
                (key, value)
                for key, value in state_dict.items()
                if any(
                    key.startswith(field_prefix) for field_prefix in text_module_fields
                )
            ]
        )

        # Loading the state weights
        self.load_state_dict(text_state)
