import dataclasses
from typing import Any, List, Sequence

import numpy as np

from mlmodule.v2.base.callbacks import BaseSaveFeaturesCallback, BaseSaveLabelsCallback
from mlmodule.v2.base.models import ModelWithLabels
from mlmodule.v2.helpers.types import NumericArrayTypes
from mlmodule.v2.helpers.utils import convert_numeric_array_like_to_numpy


@dataclasses.dataclass
class CollectFeaturesInMemory(BaseSaveFeaturesCallback[NumericArrayTypes]):
    indices: list = dataclasses.field(init=False, default_factory=list)
    features: np.ndarray = dataclasses.field(
        init=False, default_factory=lambda: np.empty(0)
    )

    def save_features(
        self, model: Any, indices: Sequence, features: NumericArrayTypes
    ) -> None:
        self.indices += list(indices)
        features_numpy = convert_numeric_array_like_to_numpy(features)

        if self.features.size > 0:
            # We stack new features
            self.features = np.vstack((self.features, features_numpy))
        else:
            # This is the fisrt feature, we replace the default value with size=0
            self.features = features_numpy


@dataclasses.dataclass
class CollectLabelsInMemory(BaseSaveLabelsCallback[NumericArrayTypes]):
    indices: list = dataclasses.field(init=False, default_factory=list)
    label_scores: np.ndarray = dataclasses.field(
        init=False, default_factory=lambda: np.empty(0)
    )
    labels: List[str] = dataclasses.field(init=False, default_factory=list)

    def save_label_scores(
        self,
        model: ModelWithLabels,
        indices: Sequence,
        labels_scores: NumericArrayTypes,
    ) -> None:
        label_set = model.get_labels()

        self.indices += list(indices)
        scores_numpy = convert_numeric_array_like_to_numpy(labels_scores)

        if self.label_scores.size > 0:
            # We stack new features
            self.label_scores = np.vstack((self.label_scores, scores_numpy))
        else:
            # This is the fisrt feature, we replace the default value with size=0
            self.label_scores = scores_numpy

        # Getting the matching label
        label_idx = scores_numpy.argmax(axis=1)
        self.labels += [label_set[idx] for idx in label_idx]
