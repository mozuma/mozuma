import dataclasses
from typing import Tuple

import torch
from tokenizers import Encoding, Tokenizer


@dataclasses.dataclass
class TokenizerTransform:
    tokenizer: Tokenizer

    def __call__(self, text: str) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        encoding: Encoding = self.tokenizer.encode(text)
        return (
            torch.LongTensor(encoding.ids),
            torch.FloatTensor(encoding.attention_mask),
        )
