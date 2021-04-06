import os
from typing import List

import torch

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.box import BBoxOutput
from mlmodule.torch.data.box import BoundingBoxDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_arcface_features_inference(torch_device: torch.device):
    arcface = ArcFaceFeatures(device=torch_device)
    arcface.load()
    mtcnn = MTCNNDetector(device=torch_device, image_size=720,
                          min_face_size=20)
    mtcnn.load()
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))

    # Load image dataset
    dataset = ImageDataset(file_names)

    # Detect faces first
    file_names, outputs = mtcnn.bulk_inference(dataset)

    # Flattening all detected faces
    bboxes: List[BBoxOutput]
    indices: List[str]
    indices, file_names, bboxes = zip(*[
        (f'{fn}_{i}', fn, bbox) for fn, bbox_list in zip(file_names, outputs) for i, bbox in enumerate(bbox_list)
    ])

    # Create a dataset for the bounding boxes
    bbox_features = BoundingBoxDataset(indices, file_names, bboxes)

    # Get face features
    indices, new_outputs = arcface.bulk_inference(
        bbox_features, data_loader_options={'batch_size': 3})
    output_by_file = dict(zip(indices, new_outputs))

    # tests
    assert output_by_file[os.path.join(base_path, 'berset1.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) > .7
    assert output_by_file[os.path.join(base_path, 'berset1.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset2.jpg_1')]) > .7
    assert output_by_file[os.path.join(base_path, 'berset2.jpg_1')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) > .7

    # different faces
    assert output_by_file[os.path.join(base_path, 'berset2.jpg_0')].dot(
        output_by_file[os.path.join(base_path, 'berset3.jpg_1')]) < .7
