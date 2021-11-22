import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class FrameOutput:
    frame_pos: int
    probability: float
    features: Optional[np.ndarray]


FrameOutputCollection = List[FrameOutput]
