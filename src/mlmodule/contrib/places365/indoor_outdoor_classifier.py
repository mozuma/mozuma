import numpy as np

from mlmodule.base import BaseMLModule
from mlmodule.labels import PlacesIOLabels

class PlacesIOClassifier(BaseMLModule):

    def __init__(self):
        super().__init__()
        self.labels_io = PlacesIOLabels()

    @classmethod
    def load(cls, fp=None):
        pass

    def dump(self, fp):
        pass

    def bulk_inference(self, data):
        """Performs inference for all the given data points

        :param data: np.ndarray(n, 365). Output of classifier trained on Places365 for n images
        :return: np.ndarray(n, ). Whether each image is predicted to be outdoors (0) or indoors (1)
        """
        # TODO: should we set k as a method parameter
        k = 10

        # As we don't care about the actual values (only which ones are the largest),
        # it doesn't matter if a softmax was computed on the output of the classifier

        # Numpy equivalent of _, idx = torch.topk(data)
        # Returns the k indices with the highest values for each row
        topk_idx = np.argpartition(data, -k, axis=1)[:, -k:]

        # Map each class in each row to either indoor (0) or outdoor (1)
        def cls_to_io(arr):
            return np.array(self.labels_io.label_list)[arr]

        topk_io = np.apply_along_axis(cls_to_io, 1, topk_idx)

        # Compute the mean number for each row
        mean_io = np.apply_along_axis(np.mean, 1, topk_io)

        # If more than half the labels indicate that the image is indoors, return indoors
        indoors = (mean_io < 0.5).astype(int)

        return indoors
