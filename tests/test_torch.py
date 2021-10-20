import tempfile
import os

import torch
import torch.nn as nn

from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.images import ImageDataset, TORCHVISION_STANDARD_IMAGE_TRANSFORMS
from mlmodule.utils import list_files_in_dir


class MyTestModuleBase(BaseTorchMLModule):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)


def assert_weights_equal(device: torch.device, model1: nn.Module, model2: nn.Module):
    model1.to(device)
    model2.to(device)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_load_dump_model(torch_device: torch.device):
    tml = MyTestModuleBase()

    with tempfile.TemporaryFile() as f:
        # Saving a file
        tml.dump(f)

        # Asserting not empty
        f.seek(0)
        assert len(f.read()) > 0

        # Loading a new model
        tml_reload = MyTestModuleBase().load(f)

    # Testing if weights are equals
    assert_weights_equal(torch_device, tml, tml_reload)


def test_image_file_dataset():
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_objects = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    dataset = ImageDataset(file_objects)
    # Trying to load an image
    assert len(dataset[10]) > 0


def test_image_file_dataset_torchvision_transform():
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_objects = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    dataset = ImageDataset(file_objects)
    dataset.add_transforms(TORCHVISION_STANDARD_IMAGE_TRANSFORMS)
    # Trying to load an image
    assert len(dataset[10]) > 0
