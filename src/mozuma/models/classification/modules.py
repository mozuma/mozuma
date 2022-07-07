from typing import List, Optional, Sequence, overload

import torch
from torch import nn

from mozuma.labels.base import LabelSet
from mozuma.predictions import BatchModelPrediction
from mozuma.states import StateType
from mozuma.torch.modules import TorchModel


@overload
def get_activation_fun_from_torch_nn(activation_name: str) -> nn.Module:
    ...


@overload
def get_activation_fun_from_torch_nn(activation_name: None) -> None:
    ...


def get_activation_fun_from_torch_nn(
    activation_name: Optional[str],
) -> Optional[nn.Module]:
    """Get the activation function from `torch.nn` and an activation name.

    See [`torch.nn`](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

    Args:
        activation_name (str | None): The name of the activation function.
            Should match the name of an attribute of `torch.nn`.

    Returns:
        torch.nn.Module | None: The module that implements the activation function
    """
    if activation_name is None:
        return None
    return getattr(nn, activation_name)()


def mlp_state_architecture(
    prefix: str, layer_dims: Sequence[int], activation: Optional[str] = None
):
    """Creates a state architecture string from a MLP parameters.

    Examples:
        `prefix=mlp, layer_dims=(10,100,50), activation=ReLU` returns `mlp-10x100x50-ReLU`

        `prefix=mlp, layer_dims=(10,100,50), activation=None` returns `mlp-10x100x50`
    """
    items = [prefix, "x".join(str(d) for d in layer_dims)]
    if activation:
        items.append(activation)
    return "-".join(items)


class LinearModuleWithActivation(nn.Module):
    """Helper class that defines a linear module with activation function"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_fun: Optional[nn.Module] = None,
        device: torch.device = None,
    ):
        super().__init__()
        seq: List[nn.Module] = [
            nn.Linear(in_features=in_dim, out_features=out_dim, device=device)
        ]
        if activation_fun:
            seq.append(activation_fun)
        self.module = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class LinearClassifierTorchModule(TorchModel[torch.Tensor, torch.Tensor]):
    """Linear classifier


    Attributes:
        in_features (int): Number of dimensions in the input
        label_set (LabelSet): The set of labels for this classifier
    """

    def __init__(
        self,
        in_features: int,
        label_set: LabelSet,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        # Input features
        self.in_features = in_features
        # Output label set
        self.label_set = label_set
        self.num_classes = len(label_set)
        # Fully connected linear module
        self.fc = nn.Linear(
            in_features=self.in_features,
            out_features=self.num_classes,
            device=self.device,
        )

    @property
    def state_type(self) -> StateType:
        return StateType(
            backend="pytorch",
            architecture=mlp_state_architecture(
                "lin", (self.in_features, self.num_classes)
            ),
            extra=(f"lbl-{self.label_set.label_set_unique_id}",),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier"""
        return self.fc(batch)

    def to_predictions(
        self, forward_output: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(label_scores=forward_output)

    def get_labels(self) -> LabelSet:
        return self.label_set


class MLPClassifierTorchModule(TorchModel[torch.Tensor, torch.Tensor]):
    """Multi-layer perceptron classifier

    Attributes:
        in_features (int): The number of dimensions in the input
        hidden_layers (Sequence[int]): A sequence of width for the hidden layers
        label_set (LabelSet): The set of labels for this classifier
        activation (str | None): Name of an activation function in the `torch.nn` module.
            For instance [`ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU).
            If `None`, no activation function is used.

    """

    def __init__(
        self,
        in_features: int,
        hidden_layers: Sequence[int],
        label_set: LabelSet,
        activation: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        # Labels
        self.label_set = label_set
        self.num_classes = len(label_set)
        # Dimension
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.layers_dim = [in_features] + list(hidden_layers) + [self.num_classes]
        # Activation
        self.activation = activation
        self.activation_fun = get_activation_fun_from_torch_nn(activation)

        # Building the sequence of Linear modules
        layers_dim_before_last = self.layers_dim[:-1]
        layers_modules = [
            LinearModuleWithActivation(
                in_dim=dim,
                out_dim=dim_next,
                activation_fun=self.activation_fun,
                device=self.device,
            )
            for dim, dim_next in zip(
                layers_dim_before_last[:-1], layers_dim_before_last[1:]
            )
        ] + [  # Last layer does not have an activation function
            LinearModuleWithActivation(
                in_dim=self.layers_dim[-2],
                out_dim=self.layers_dim[-1],
                device=self.device,
            )
        ]
        self.mlp = nn.Sequential(*layers_modules)

    @property
    def state_type(self) -> StateType:
        return StateType(
            backend="pytorch",
            architecture=mlp_state_architecture(
                "mlp", self.layers_dim, activation=self.activation
            ),
            extra=(f"lbl-{self.label_set.label_set_unique_id}",),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.mlp(batch)

    def to_predictions(
        self, forward_output: torch.Tensor
    ) -> BatchModelPrediction[torch.Tensor]:
        return BatchModelPrediction(label_scores=forward_output)

    def get_labels(self) -> LabelSet:
        return self.label_set
