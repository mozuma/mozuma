from typing import Dict, List, Tuple, Union

import numpy as np
from PIL.Image import Image
from torch import Tensor

StateDict = Dict[str, Tensor]

MetricValue = Union[str, int, float]
Metrics = Dict[str, MetricValue]

ImageDatasetType = Union[Image, np.ndarray]

# Videos
FrameIdxType = int
FrameSequenceType = Tuple[List[FrameIdxType], List[Image]]
