import os
from typing import List

import torch
import numpy as np

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.magface import MagFaceFeatures
from mlmodule.box import BBoxOutput
from mlmodule.torch.data.box import BoundingBoxDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

FACE_DISTANCE_THRESHOLD = 0.5


def _face_features_for_folder(torch_device: torch.device, folder, **opts):
    magface = MagFaceFeatures(device=torch_device)
    magface.load()
    mtcnn = MTCNNDetector(device=torch_device, min_face_size=20)
    mtcnn.load()
    file_names = list_files_in_dir(folder, allowed_extensions=('jpg', 'png'))

    # Load image dataset
    dataset = ImageDataset(file_names)

    # Detect faces first
    d_indices, outputs = mtcnn.bulk_inference(dataset)

    # Flattening all detected faces
    bboxes: List[BBoxOutput]
    indices: List[str]
    indices, file_names, bboxes = zip(*[
        (f'{fn}_{i}', fn, bbox) for fn, bbox_list in zip(d_indices, outputs) for i, bbox in enumerate(bbox_list)
    ])

    # Create a dataset for the bounding boxes
    bbox_features = BoundingBoxDataset(indices, file_names, bboxes)

    # Get face features
    return (d_indices, outputs), magface.bulk_inference(
        bbox_features, data_loader_options={'batch_size': 3}, **opts)


def test_magface_features_inference(torch_device: torch.device):
    base_path = os.path.join("tests", "fixtures", "berset")
    _, (indices, new_outputs) = _face_features_for_folder(torch_device, base_path)
    normalized_features = new_outputs / np.linalg.norm(new_outputs, axis=1, keepdims=True)
    output_by_file = dict(zip(indices, normalized_features))

    # tests
    assert output_by_file[os.path.join(base_path, 'berset1.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) > FACE_DISTANCE_THRESHOLD
    assert output_by_file[os.path.join(base_path, 'berset1.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset2.jpg_1')]) > FACE_DISTANCE_THRESHOLD
    assert output_by_file[os.path.join(base_path, 'berset2.jpg_1')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) > FACE_DISTANCE_THRESHOLD

    # different faces
    assert output_by_file[os.path.join(base_path, 'berset2.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) < FACE_DISTANCE_THRESHOLD


def test_bad_quality_face_filter():
    (detect_i, detect_box), (indices, new_outputs) = _face_features_for_folder(
        torch.device('cpu'), os.path.join("tests", "fixtures", "faces")
    )
    # office_blur has 2 visible faces
    assert len([i for i in indices if 'office_blur' in i]) == 1
    # But 3 detected faces
    assert sum([len(b) for i, b in zip(detect_i, detect_box) if 'office_blur' in i]) == 3
