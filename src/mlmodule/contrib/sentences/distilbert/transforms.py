import dataclasses
from typing import Tuple

import torch
from tokenizers import Encoding, Tokenizer


@dataclasses.dataclass
class TokenizerTransform:
    tokenizer: Tokenizer

    def __call__(self, text: str) -> Tuple[torch.IntTensor, torch.IntTensor]:
        encoding: Encoding = self.tokenizer.encode(text)
        return (torch.IntTensor(encoding.ids), torch.IntTensor(encoding.attention_mask))
