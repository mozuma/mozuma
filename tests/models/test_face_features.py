import itertools
import os
from typing import Dict, List, Optional, Sequence, Type, Union

import numpy as np
import pytest
import torch

from mlmodule.callbacks.memory import (
    CollectBoundingBoxesInMemory,
    CollectFeaturesInMemory,
)
from mlmodule.helpers.files import list_files_in_dir
from mlmodule.models.arcface.modules import TorchArcFaceModule
from mlmodule.models.arcface.pretrained import torch_arcface_insight_face
from mlmodule.models.magface.modules import TorchMagFaceModule
from mlmodule.models.mtcnn.modules import TorchMTCNNModule
from mlmodule.states import StateKey
from mlmodule.stores import Store
from mlmodule.torch.datasets import (
    ImageBoundingBoxDataset,
    ImageDataset,
    LocalBinaryFilesDataset,
)
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.runners import TorchInferenceRunner

_FaceModelType = Union[Type[TorchMagFaceModule], Type[TorchArcFaceModule]]

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
    face_module: _FaceModelType,
    torch_device: torch.device,
    folder: str,
    bounding_boxes: Optional[CollectBoundingBoxesInMemory] = None,
    remove_bad_faces: bool = False,
) -> CollectFeaturesInMemory:
    # Face detection
    bounding_boxes = bounding_boxes or _face_detection_for_folder(torch_device, folder)

    # Loading model with pre-trained state
    model = face_module(device=torch_device, remove_bad_faces=remove_bad_faces)

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


@pytest.mark.parametrize(
    "face_module",
    [TorchMagFaceModule, torch_arcface_insight_face],
    ids=["magface", "arcface"],
)
def test_face_features_inference(
    torch_device: torch.device, face_module: _FaceModelType
):
    base_path = os.path.join("tests", "fixtures", "berset")
    feature = _face_features_for_folder(face_module, torch_device, base_path)
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


@pytest.mark.parametrize(
    "remove_bad_faces", [True, False], ids=["remove-bad", "keep-bad"]
)
@pytest.mark.parametrize(
    "face_module,n_good_faces",
    [(TorchMagFaceModule, 1), (torch_arcface_insight_face, 3)],
    ids=["magface", "arcface"],
)
def test_bad_quality_face_filter(
    remove_bad_faces: bool,
    face_module: _FaceModelType,
    n_good_faces: int,
):
    # Defining variables
    base_path = os.path.join("tests", "fixtures", "faces")
    device = torch.device("cpu")
    blurry_picture_name = os.path.join(base_path, "office_blur.jpg")

    # Getting face detection output
    bb = _face_detection_for_folder(device, base_path)
    # Getting face features
    ff = _face_features_for_folder(
        face_module,
        device,
        base_path,
        bb,
        remove_bad_faces=remove_bad_faces,
    )

    # COunting the number of bounding boxes for the blurry picture
    detect_index_blur = bb.indices.index(blurry_picture_name)
    count_bbox = len(bb.bounding_boxes[detect_index_blur].bounding_boxes)

    # Counting the number of features without NaN for the blurry_picture_name
    features_index_blur = [
        box_index
        for (path, box_index), nan_features in zip(
            ff.indices, np.isnan(ff.features).all(axis=1)
        )
        if path == blurry_picture_name and not nan_features
    ]
    count_features = len(features_index_blur)

    if remove_bad_faces:
        # Blurry face picture visible faces
        assert count_features == n_good_faces
    else:
        assert count_features == count_bbox
    # But more detected faces
    assert count_bbox > n_good_faces
