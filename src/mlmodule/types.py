from typing import Dict, Union

from torch import Tensor


StateDict = Dict[str, Tensor]

MetricValue = Union[str, int, float]
Metrics = Dict[str, MetricValue]
