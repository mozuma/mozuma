import dataclasses
from typing import Sequence

import numpy as np

from mlmodule.v2.base.callbacks import BaseSaveFeaturesCallback
from mlmodule.v2.helpers.types import NumericArrayTypes
from mlmodule.v2.helpers.utils import convert_numeric_array_like_to_numpy


@dataclasses.dataclass
class CollectFeaturesInMemory(BaseSaveFeaturesCallback[NumericArrayTypes]):
    indices: list = dataclasses.field(init=False, default_factory=list)
    features: np.ndarray = dataclasses.field(
        init=False, default_factory=lambda: np.empty(0)
    )

    def save_features(self, indices: Sequence, features: NumericArrayTypes) -> None:
        self.indices += list(indices)
        features_numpy = convert_numeric_array_like_to_numpy(features)

        if self.features.size > 0:
            # We stack new features
            self.features = np.vstack((self.features, features_numpy))
        else:
            # This is the fisrt feature, we replace the default value with size=0
            self.features = features_numpy
