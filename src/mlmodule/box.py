"""
This file contains classes to help organise complex model outputs such as bounding boxes
"""
import dataclasses
from typing import Tuple, NamedTuple, Optional, List

import numpy as np


@dataclasses.dataclass
class BBoxPoint:
    """Bounding box point coordinates"""
    x: float
    y: float


@dataclasses.dataclass
class BBoxOutput:
    """Bounding box model output"""
    bounding_box: Tuple[BBoxPoint, BBoxPoint]   # Point pair
    probability: float  # Confidence of the bounding box
    features: Optional[np.ndarray] = None


BBoxCollection = List[BBoxOutput]
