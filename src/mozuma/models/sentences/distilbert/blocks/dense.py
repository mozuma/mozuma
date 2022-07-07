from typing import Dict

from torch import Tensor, nn


class Dense(nn.Module):
    """Feed-forward function with  activiation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer.
    Can be used to generate deep averaging networs (DAN).

    :param in_features: Size of the input dimension
    :param out_features: Output size
    :param bias: Add a bias vector
    :param activation_function: Pytorch activation function applied on output
    :param init_weight: Initial value for the matrix of the linear layer
    :param init_bias: Initial value for the bias of the linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function=nn.Tanh(),
        init_weight: Tensor = None,
        init_bias: Tensor = None,
    ):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: Dict[str, Tensor]):
        features.update(
            {
                "sentence_embedding": self.activation_function(
                    self.linear(features["sentence_embedding"])
                )
            }
        )
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def __repr__(self):
        return "Dense({})".format(self.get_config_dict())
