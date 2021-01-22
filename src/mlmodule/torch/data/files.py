from mlmodule.torch.data.base import BaseIndexedDataset


class FilesDataset(BaseIndexedDataset):

    def __init__(self, file_list, mode="rb"):
        """
        :param file_list: Must be a string
        """
        super().__init__(file_list)
        self.mode = mode
        self.add_transforms([self.open])

    def open(self, item):
        with open(item, mode=self.mode) as f:
            return f
