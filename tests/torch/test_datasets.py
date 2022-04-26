import os
from typing import List

import numpy as np
import pytest
from torchvision.transforms import ToTensor

from mlmodule.v2.base.predictions import BatchBoundingBoxesPrediction
from mlmodule.v2.torch.datasets import (
    ImageBoundingBoxDataset,
    ImageDataset,
    LocalBinaryFilesDataset,
    TorchTrainingDataset,
)


@pytest.mark.parametrize("resize", [True, False], ids=["resize", "no-resize"])
def test_image_dataset(cats_and_dogs_images: List[str], resize: bool):
    dataset = ImageDataset(
        binary_files_dataset=LocalBinaryFilesDataset(paths=cats_and_dogs_images),
        resize_image_size=(120, 120) if resize else None,
    )

    # Checking length
    assert len(dataset) == len(cats_and_dogs_images)

    # Checking indices
    assert cats_and_dogs_images == [
        dataset.getitem_indices(i) for i in range(len(dataset))
    ]

    # Checking returned data
    index, image = dataset[0]
    assert index == cats_and_dogs_images[0]
    if resize:
        assert image.size == (120, 120)
    # Make sure that the image can be read as an array
    assert np.array(image).shape == image.size[::-1] + (3,)


@pytest.mark.parametrize("to_rgb", (True, False))
def test_image_dataset_rgba(to_rgb: bool):
    dataset = ImageDataset(
        binary_files_dataset=LocalBinaryFilesDataset(
            paths=[os.path.join("tests", "fixtures", "rgba", "alpha.png")]
        ),
        mode="RGB" if to_rgb else None,
    )

    # Getting the image
    _, image = dataset[0]
    image_shape = ToTensor()(image).shape
    assert len(image_shape) == 3
    if to_rgb:
        assert image_shape[:1] == (3,)
    else:
        assert image_shape[:1] == (4,)


@pytest.mark.parametrize("crop_image", [True, False], ids=["crop", "no-crop"])
def test_bounding_box_dataset(cats_and_dogs_images: List[str], crop_image: bool):
    # Dataset with 2 images and 3 bounding boxes
    dataset = ImageBoundingBoxDataset(
        image_dataset=ImageDataset(LocalBinaryFilesDataset(cats_and_dogs_images[:2])),
        bounding_boxes=[
            BatchBoundingBoxesPrediction(
                bounding_boxes=np.array([[0, 0, 10, 10], [5, 0, 25, 25]])
            ),
            BatchBoundingBoxesPrediction(bounding_boxes=np.array([[0, 0, 10, 10]])),
        ],
        crop_image=crop_image,
    )

    assert len(dataset) == 2 + 1  # 2 for the first image and 1 for the second image

    # Getting the first sample
    (image_path, bbox_index), (crop, bbox) = dataset[0]
    assert image_path == cats_and_dogs_images[0]
    assert bbox_index == 0
    if crop_image:
        assert crop.size == (10, 10)
    else:
        assert crop.size != (10, 10)
    # Make sure that the image can be read as an array
    assert np.array(crop).shape == crop.size[::-1] + (3,)
    np.testing.assert_equal(bbox.bounding_boxes, np.array([[0, 0, 10, 10]]))

    # Getting the first sample, second box
    (image_path, bbox_index), (crop, bbox) = dataset[1]
    assert image_path == cats_and_dogs_images[0]
    assert bbox_index == 1
    if crop_image:
        assert crop.size == (20, 25)
    else:
        assert crop.size != (20, 25)
    np.testing.assert_equal(bbox.bounding_boxes, np.array([[5, 0, 25, 25]]))


def test_training_dataset(cats_and_dogs_images):
    labels = [f"label_{i}" for i in range(len(cats_and_dogs_images))]

    image_dataset = ImageDataset(
        binary_files_dataset=LocalBinaryFilesDataset(paths=cats_and_dogs_images)
    )
    dataset = TorchTrainingDataset(dataset=image_dataset, target_labels=labels)

    # Checking length
    assert len(dataset) == len(cats_and_dogs_images)

    # Checking class (self.classes and self.classes_to_idx)
    # (since in this test all labels are unique the number of classes is the same as the size)
    assert all(item[1][1] == i for i, item in enumerate(dataset))
