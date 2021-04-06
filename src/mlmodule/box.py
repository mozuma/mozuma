"""
This file contains classes to help organise complex model outputs such as bounding boxes
"""
from typing import Tuple, NamedTuple, Optional, List

import numpy as np


class BBoxPoint(NamedTuple):
    """Bounding box point coordinates"""
    x: float
    y: float


class BBoxOutput(NamedTuple):
    """Bounding box model output"""
    bounding_box: Tuple[BBoxPoint, BBoxPoint]   # Point pair
    probability: float  # Confidence of the bounding box
    features: Optional[np.ndarray] = None


BBoxCollection = List[BBoxOutput]
