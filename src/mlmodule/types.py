from typing import Dict, Union

from PIL import Image
import numpy as np
from torch import Tensor


StateDict = Dict[str, Tensor]

MetricValue = Union[str, int, float]
Metrics = Dict[str, MetricValue]

ImageDatasetType = Union[Image.Image, np.ndarray]
