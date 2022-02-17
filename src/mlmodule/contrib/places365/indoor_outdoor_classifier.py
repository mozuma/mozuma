import numpy as np
from scipy.special import softmax

from mlmodule.base import BaseMLModule
from mlmodule.labels.base import LabelSet
from mlmodule.labels.places_io import PLACES_IN_OUT_DOOR, PLACES_IO_LABELS


class PlacesIOClassifier(BaseMLModule):
    def __init__(self, k=10, **_):
        super().__init__()
        self.labels_io = PLACES_IO_LABELS
        self.k = k

    def bulk_inference(self, data, **_):
        """Performs inference for all the given data points

        :param data: np.ndarray(n, 365). Output of classifier trained on Places365 for n images
        :return: np.ndarray(n, 2). Each image is assigned a probability of
         being indoor in position 0 and outdoor in position 1
        """

        # As we don't care about the actual values (only which ones are the largest),
        # it doesn't matter if a softmax was computed on the output of the classifier

        # Numpy equivalent of _, idx = torch.topk(data)
        # Returns the k indices with the highest values for each row
        idx, probs = list(zip(*data))
        topk_idx = np.argpartition(softmax(np.array(probs), axis=1), -self.k, axis=1)[
            :, -self.k :
        ]

        # Map each class in each row to either indoor (0) or outdoor (1)
        def cls_to_io(arr):
            return np.array(self.labels_io.label_list)[arr]

        topk_io = np.apply_along_axis(cls_to_io, 1, topk_idx)

        # Compute the mean number for each row
        mean_io = np.apply_along_axis(np.mean, 1, topk_io)

        return idx, np.vstack((1 - mean_io, mean_io)).T

    def get_labels(self) -> LabelSet:
        return PLACES_IN_OUT_DOOR
