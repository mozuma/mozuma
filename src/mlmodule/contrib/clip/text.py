__all__ = (
    'CLIPTextEncoder',
    'CLIPResNet50TextEncoder',
    'CLIPResNet101TextEncoder',
    'CLIPResNet50x4TextEncoder',
    'CLIPViTB32TextEncoder'
)

from typing import Optional

import clip
import torch

from mlmodule.contrib.clip.base import BaseCLIPModule


def tokenize_single(text: str) -> torch.LongTensor:
    return clip.tokenize(text)[0]


class CLIPTextEncoder(BaseCLIPModule):

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)

        clip_module = self._get_clip_module()

        # Populating with text encoder attributes
        self.context_length = clip_module.context_length
        self.vocab_size = clip_module.vocab_size
        self.token_embedding = clip_module.token_embedding
        self.positional_embedding = clip_module.positional_embedding
        self.transformer = clip_module.transformer
        self.ln_final = clip_module.ln_final
        self.text_projection = clip_module.text_projection

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def get_dataset_transforms(self):
        return [tokenize_single]


class CLIPResNet50TextEncoder(CLIPTextEncoder):
    clip_model_name = "RN50"


class CLIPResNet101TextEncoder(CLIPTextEncoder):
    clip_model_name = "RN101"


class CLIPResNet50x4TextEncoder(CLIPTextEncoder):
    clip_model_name = "RN50x4"


class CLIPViTB32TextEncoder(CLIPTextEncoder):
    clip_model_name = "ViT-B/32"
