import os
from typing import List, Tuple

import torch
import numpy as np

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.magface import MagFaceFeatures
from mlmodule.box import BBoxCollection, BBoxOutput
from mlmodule.torch.data.box import BoundingBoxDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

FACE_DISTANCE_THRESHOLD = 0.5


def _face_features_for_folder(
        torch_device: torch.device, folder, **opts
) -> Tuple[
        Tuple[List[str], List[BBoxCollection]],
        Tuple[List[str], np.ndarray]
]:
    magface = MagFaceFeatures[str](device=torch_device)
    magface.load()
    mtcnn = MTCNNDetector[str](device=torch_device, min_face_size=20)
    mtcnn.load()
    file_names = list_files_in_dir(folder, allowed_extensions=('jpg', 'png'))

    # Load image dataset
    dataset = ImageDataset(file_names)

    # Detect faces first
    ret = mtcnn.bulk_inference(dataset)
    assert ret is not None
    d_indices, outputs = ret

    # Flattening all detected faces
    indices: List[str] = []
    a_file_names: List[str] = []
    bboxes: List[BBoxOutput] = []
    for fn, bbox_list in zip(d_indices, outputs):
        for i, bbox in enumerate(bbox_list):
            indices.append(f'{fn}_{i}')
            a_file_names.append(fn)
            bboxes.append(bbox)

    # Create a dataset for the bounding boxes
    bbox_features = BoundingBoxDataset[str](indices, a_file_names, bboxes)
    magface_ret = magface.bulk_inference(
        bbox_features, data_loader_options={'batch_size': 3}, **opts
    )
    assert magface_ret is not None

    # Get face features
    return (d_indices, outputs), magface_ret


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
    (detect_i, detect_box), (indices, _) = _face_features_for_folder(
        torch.device('cpu'), os.path.join("tests", "fixtures", "faces")
    )
    # office_blur has 2 visible faces
    assert len([i for i in indices if 'office_blur' in i]) == 1
    # But 3 detected faces
    assert sum([len(b) for i, b in zip(detect_i, detect_box) if 'office_blur' in i]) == 3
