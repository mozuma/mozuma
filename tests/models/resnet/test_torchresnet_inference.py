import os
from typing import List, cast
from unittest import mock

import pytest
import torch

from mozuma.callbacks.memory import CollectLabelsInMemory
from mozuma.models.resnet.modules import TorchResNetModule, TorchResNetTrainingMode
from mozuma.models.resnet.pretrained import torch_resnet_imagenet
from mozuma.torch.datasets import ImageDataset, LocalBinaryFilesDataset
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner


def test_resnet_cats_dogs(cats_and_dogs_images: List[str], torch_device: torch.device):
    """Test ResNet classification of cats and dogs images"""
    # Creating a dataset
    dataset_dir = os.path.dirname(cats_and_dogs_images[0])
    dataset = ImageDataset(LocalBinaryFilesDataset(cats_and_dogs_images))

    # Loading model
    model = torch_resnet_imagenet("resnet18")

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


@pytest.mark.parametrize(
    "training_mode",
    [None, TorchResNetTrainingMode.features, TorchResNetTrainingMode.labels],
    ids=["no_training_mode", "training_mode_features", "training_mode_labels"],
)
def test_resnet_forward_output(training_mode):
    # Loading model
    model = TorchResNetModule("resnet18", training_mode=training_mode)

    input_example = torch.zeros([1])
    expected = (
        torch.zeros([1]),
        torch.zeros([2]),
    )  # features, labels_scores

    with mock.patch.object(TorchResNetModule, "forward_features") as ff:
        ff.return_value = expected[0]

        with mock.patch.object(TorchResNetModule, "forward_classifier") as fc:
            fc.return_value = expected[1]

            # call forward pass
            output = model(input_example)

            # Check forward() returns according to training_mode
            if not training_mode:
                assert isinstance(output, tuple) and len(output) == 2

                assert torch.equal(output[0], expected[0])
                assert torch.equal(output[1], expected[1])
            elif training_mode == TorchResNetTrainingMode.features:
                assert isinstance(output, torch.Tensor)

                assert torch.equal(output, expected[0])
            elif training_mode == TorchResNetTrainingMode.labels:
                assert isinstance(output, torch.Tensor)

                assert torch.equal(output, expected[1])

            # Check to_predictions()
            predictions = model.to_predictions(output)

            if not training_mode:
                assert (
                    predictions.features is not None
                    and predictions.label_scores is not None
                )

            elif training_mode == TorchResNetTrainingMode.features:
                assert (
                    predictions.features is not None
                    and predictions.label_scores is None
                )

            elif training_mode == TorchResNetTrainingMode.labels:
                assert (
                    predictions.features is None
                    and predictions.label_scores is not None
                )
