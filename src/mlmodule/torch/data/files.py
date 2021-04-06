from pathlib import Path
from typing import IO, Union

from mlmodule.torch.data.base import IndexedDataset


ReadablePathType = Union[str, Path, IO]


class FilesDataset(IndexedDataset):

    def __init__(self, file_list, mode="rb"):
        """
        :param file_list: Must be a string
        """
        # Setting index to the filename
        super().__init__(file_list, file_list)
        self.mode = mode
        self.add_transforms([self.open])

    def open(self, item):
        with open(item, mode=self.mode) as f:
            return f
