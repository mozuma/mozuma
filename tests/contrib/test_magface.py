import itertools
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mlmodule.contrib.magface.modules import TorchMagFaceModule
from mlmodule.contrib.mtcnn.modules import TorchMTCNNModule
from mlmodule.utils import list_files_in_dir
from mlmodule.v2.helpers.callbacks import (
    CollectBoundingBoxesInMemory,
    CollectFeaturesInMemory,
)
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.torch.datasets import (
    ImageBoundingBoxDataset,
    ImageDataset,
    LocalBinaryFilesDataset,
)
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner

FACE_DISTANCE_THRESHOLD = 0.5


def _face_detection_for_folder(
    torch_device: torch.device, folder: str
) -> CollectBoundingBoxesInMemory:
    # Loading model with pre-trained state
    model = TorchMTCNNModule(device=torch_device)
    Store().load(model, StateKey(model.state_type, training_id="facenet"))

    # Dataset from images in a folder
    file_names = list_files_in_dir(folder, allowed_extensions=("jpg", "png"))

    # Load image dataset
    dataset = ImageDataset(LocalBinaryFilesDataset(file_names))

    # Callbacks
    bb = CollectBoundingBoxesInMemory()

    # Runner
    runner = TorchInferenceRunner(
        model=model,
        dataset=dataset,
        callbacks=[bb],
        options=TorchRunnerOptions(device=torch_device),
    )
    runner.run()

    return bb


def _face_features_for_folder(
    torch_device: torch.device,
    folder: str,
    bounding_boxes: Optional[CollectBoundingBoxesInMemory] = None,
) -> CollectFeaturesInMemory:
    # Face detection
    bounding_boxes = bounding_boxes or _face_detection_for_folder(torch_device, folder)

    # Loading model with pre-trained state
    model = TorchMagFaceModule(device=torch_device)
    Store().load(model, StateKey(model.state_type, "magface"))

    # Dataset
    dataset = ImageBoundingBoxDataset(
        image_dataset=ImageDataset(LocalBinaryFilesDataset(bounding_boxes.indices)),
        bounding_boxes=bounding_boxes.bounding_boxes,
    )

    # Callbacks
    ff = CollectFeaturesInMemory()

    # Runner
    runner = TorchInferenceRunner(
        model=model,
        dataset=dataset,
        callbacks=[ff],
        options=TorchRunnerOptions(device=torch_device),
    )
    runner.run()

    return ff


def _count_matching_face_features(f1: Sequence[np.ndarray], f2: Sequence[np.ndarray]):
    count = 0
    for ff1, ff2 in itertools.product(f1, f2):
        if ff1.dot(ff2) > FACE_DISTANCE_THRESHOLD:
            count += 1
    return count


def test_magface_features_inference(torch_device: torch.device):
    base_path = os.path.join("tests", "fixtures", "berset")
    feature = _face_features_for_folder(torch_device, base_path)
    normalized_features: np.ndarray = feature.features / np.linalg.norm(
        feature.features, axis=1, keepdims=True
    )
    output_by_file: Dict[str, List[np.ndarray]] = {}
    for (path, _), box_features in zip(feature.indices, normalized_features):
        output_by_file.setdefault(path, [])
        output_by_file[path].append(box_features)

    # tests
    assert (
        _count_matching_face_features(
            output_by_file[os.path.join(base_path, "berset1.jpg")],
            output_by_file[os.path.join(base_path, "berset3.jpg")],
        )
        == 1
    )
    assert (
        _count_matching_face_features(
            output_by_file[os.path.join(base_path, "berset1.jpg")],
            output_by_file[os.path.join(base_path, "berset2.jpg")],
        )
        == 1
    )
    assert (
        _count_matching_face_features(
            output_by_file[os.path.join(base_path, "berset2.jpg")],
            output_by_file[os.path.join(base_path, "berset3.jpg")],
        )
        == 1
    )


def test_bad_quality_face_filter():
    # Defining variables
    base_path = os.path.join("tests", "fixtures", "faces")
    device = torch.device("cpu")
    blurry_picture_name = os.path.join(base_path, "office_blur.jpg")

    # Getting face detection output
    bb = _face_detection_for_folder(device, base_path)
    # Getting face features
    ff = _face_features_for_folder(device, base_path, bb)

    # COunting the number of bounding boxes for the blurry picture
    detect_index_blur = bb.indices.index(blurry_picture_name)
    count_bbox = len(bb.bounding_boxes[detect_index_blur].bounding_boxes)

    # Counting the number of features for the same picture
    features_index_blur = [
        box_index for path, box_index in ff.indices if path == blurry_picture_name
    ]
    count_features = len(features_index_blur)

    # Blurry face picture has 1 visible faces
    assert count_features == 1
    # But more detected faces
    assert count_bbox > 1
