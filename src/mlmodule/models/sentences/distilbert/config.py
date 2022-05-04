import dataclasses


@dataclasses.dataclass
class DistilBertConfig:
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    sinusoidal_pos_embds: bool = False
    n_layers: int = 6
    n_heads: int = 12
    dim: int = 768
    hidden_dim: int = 4 * 768
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    initializer_range: float = 0.02
    qa_dropout: float = 0.1
    seq_classif_dropout: float = 0.2
    pad_token_id: int = 0
    output_attentions: bool = False
    output_hidden_states: bool = False
    chunk_size_feed_forward: int = 0

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    @property
    def num_attention_heads(self) -> int:
        return self.n_heads

    @property
    def hidden_size(self) -> int:
        return self.dim
