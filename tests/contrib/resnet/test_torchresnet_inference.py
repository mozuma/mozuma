import os
from typing import List, cast

import torch

from mlmodule.contrib.resnet.modules import TorchResNetModule
from mlmodule.v2.helpers.callbacks import CollectLabelsInMemory
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.torch.datasets import OpenImageFileDataset
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


def test_resnet_cats_dogs(cats_and_dogs_images: List[str], torch_device: torch.device):
    """Test ResNet classification of cats and dogs images"""
    # Creating a dataset
    dataset_dir = os.path.dirname(cats_and_dogs_images[0])
    dataset = OpenImageFileDataset(cats_and_dogs_images)

    # Loading model
    model = TorchResNetModule("resnet18")
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
    assert "cat" in labels_dict[os.path.join(dataset_dir, "cat_921.jpg")].lower()
    # Dog sample
    dog_file = os.path.join(dataset_dir, "dog_900.jpg")
    assert (
        "pointer" in labels_dict[dog_file].lower()
        or "labrador" in labels_dict[dog_file].lower()
    )
