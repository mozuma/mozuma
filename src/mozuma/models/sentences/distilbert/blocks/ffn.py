from torch import nn

from mozuma.models.sentences.distilbert.config import DistilBertConfig
from mozuma.models.sentences.distilbert.utils import apply_chunking_to_forward


class FFN(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in [
            "relu",
            "gelu",
        ], f"activation ({config.activation}) must be in ['relu', 'gelu']"
        self.activation = (
            nn.functional.gelu if config.activation == "gelu" else nn.ReLU()
        )

    def forward(self, input):
        return apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input
        )

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
