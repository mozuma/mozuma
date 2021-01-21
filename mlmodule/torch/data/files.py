from torch.utils.data.dataset import Dataset
from mlmodule.torch.mixins import TorchDatasetTransformsMixin


class FilesDataset(Dataset, TorchDatasetTransformsMixin):

    def __init__(self, file_list, mode="rb"):
        """
        :param file_list: Must be a string
        """
        self.file_list = file_list
        self.mode = mode
        self.transforms = []

    def __getitem__(self, item):
        with open(self.file_list[item], mode=self.mode) as c_file:
            return item, self.apply_transforms(c_file)

    def __len__(self):
        return len(self.file_list)
