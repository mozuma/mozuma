import os
import pathlib
from typing import List, cast

import torch

from mlmodule.callbacks.memory import CollectLabelsInMemory
from mlmodule.models.densenet.pretrained import (
    torch_densenet_imagenet,
    torch_densenet_places365,
)
from mlmodule.torch.datasets import ImageDataset, LocalBinaryFilesDataset
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.runners import TorchInferenceRunner


def test_densenet_cats_dogs(
    cats_and_dogs_images: List[str], torch_device: torch.device
):
    """Test DenseNet classification of cats and dogs images"""
    # Creating a dataset
    dataset_dir = os.path.dirname(cats_and_dogs_images[0])
    dataset = ImageDataset(LocalBinaryFilesDataset(cats_and_dogs_images))

    # Loading model
    model = torch_densenet_imagenet("densenet161")

    # Inference runner for Torch model
    labels = CollectLabelsInMemory()
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=model,
        callbacks=[labels],
        options=TorchRunnerOptions(device=torch_device),
    )
    runner.run()

    # Checking the resulting labels
    labels_dict = dict(zip(cast(List[str], labels.indices), labels.labels))
    # Cat sample
    assert "cat" in labels_dict[os.path.join(dataset_dir, "cat_930.jpg")].lower()
    # Dog sample
    dog_file = os.path.join(dataset_dir, "dog_900.jpg")
    assert (
        "pointer" in labels_dict[dog_file].lower()
        or "labrador" in labels_dict[dog_file].lower()
    )


def test_densenet_places365(torch_device: torch.device):
    """Test DenseNet classification of cats and dogs images"""
    # Creating a dataset
    office_picture = str(pathlib.Path("tests", "fixtures", "berset", "berset3.jpg"))
    outdoor_picture = str(pathlib.Path("tests", "fixtures", "objects", "soldiers.jpg"))
    dataset = ImageDataset(LocalBinaryFilesDataset([office_picture, outdoor_picture]))

    # Loading model
    model = torch_densenet_places365()

    # Inference runner for Torch model
    labels = CollectLabelsInMemory()
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=model,
        callbacks=[labels],
        options=TorchRunnerOptions(device=torch_device),
    )
    runner.run()

    # Checking the resulting labels
    labels_dict = dict(zip(cast(List[str], labels.indices), labels.labels))
    print(labels_dict)
    # Office sample
    assert "legislative chamber" == labels_dict[office_picture].lower()
    # Dog sample
    assert "army base" == labels_dict[outdoor_picture].lower()
