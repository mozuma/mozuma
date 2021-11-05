from typing import Dict, Union, List, Tuple

from PIL.Image import Image
import numpy as np
from torch import Tensor


StateDict = Dict[str, Tensor]

MetricValue = Union[str, int, float]
Metrics = Dict[str, MetricValue]

ImageDatasetType = Union[Image, np.ndarray]

# Videos
FrameIdxType = int
FrameSequenceType = Tuple[List[FrameIdxType], List[Image]]
