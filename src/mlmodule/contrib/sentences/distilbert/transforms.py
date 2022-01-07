import dataclasses
from typing import Tuple

import torch
from tokenizers import Encoding, Tokenizer


@dataclasses.dataclass
class TokenizerTransform:
    tokenizer: Tokenizer

    def __call__(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding: Encoding = self.tokenizer.encode(text)
        return (torch.Tensor(encoding.ids), torch.Tensor(encoding.attention_mask))
