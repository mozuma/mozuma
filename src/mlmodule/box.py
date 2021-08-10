"""
This file contains classes to help organise complex model outputs such as bounding boxes
"""
import dataclasses
from typing import Tuple, Optional, List

import numpy as np


@dataclasses.dataclass
class BBoxPoint:
    """Bounding box point coordinates"""
    x: float
    y: float

    def __getitem__(self, item: int) -> float:
        """Makes the BBoxPoint behave as a tuple with self[0] = x and self[1] = y

        Raises:
            IndexError: if `item` > 1
        """
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError(f'Index {item} is out of range for BBoxPoint')

    def __len__(self) -> int:
        """The length of a BBoxPoint is fixed"""
        return 2


@dataclasses.dataclass
class BBoxOutput:
    """Bounding box model output"""
    bounding_box: Tuple[BBoxPoint, BBoxPoint]   # Point pair
    probability: float  # Confidence of the bounding box
    features: Optional[np.ndarray] = None


BBoxCollection = List[BBoxOutput]
