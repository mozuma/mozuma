import os
import pathlib
from typing import List, cast

import torch

from mlmodule.labels.places import PLACES_LABELS
from mlmodule.models.densenet.modules import TorchDenseNetModule
from mlmodule.v2.helpers.callbacks import CollectLabelsInMemory
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.torch.datasets import ImageDataset, LocalBinaryFilesDataset
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


def test_densenet_cats_dogs(
    cats_and_dogs_images: List[str], torch_device: torch.device
):
    """Test DenseNet classification of cats and dogs images"""
    # Creating a dataset
    dataset_dir = os.path.dirname(cats_and_dogs_images[0])
    dataset = ImageDataset(LocalBinaryFilesDataset(cats_and_dogs_images))

    # Loading model
    model = TorchDenseNetModule("densenet161")
    # Pre-trained state
    Store().load(
        model, state_key=StateKey(state_type=model.state_type, training_id="imagenet")
    )

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
    model = TorchDenseNetModule("densenet161", label_set=PLACES_LABELS)
    # Pre-trained state
    Store().load(
        model, state_key=StateKey(state_type=model.state_type, training_id="places365")
    )

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
