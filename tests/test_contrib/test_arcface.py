from conftest import device_parametrize
import os
import torch

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.torch.data.images import ImageDataset
from mlmodule.torch.data.faces import FaceDataset
from mlmodule.utils import list_files_in_dir


@device_parametrize
def test_arcface_features_inference(device):
    arcface = ArcFaceFeatures(device=device).load()
    mtcnn = MTCNNDetector(device=device, image_size=720,
                          min_face_size=20).load()
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))

    # Load image dataset
    dataset = ImageDataset(file_names)

    # Detect faces first
    file_names, outputs = mtcnn.bulk_inference(dataset)

    # Load dataset with face descriptors
    face_dataset = FaceDataset(file_names, outputs)

    # Get face features
    file_names, new_outputs = arcface.bulk_inference(face_dataset)
    output_by_file = dict(zip(file_names, new_outputs))

    # tests
    assert torch.any(torch.matmul(
        output_by_file[os.path.join(base_path, 'berset1.jpg')].features,
        output_by_file[os.path.join(base_path, 'berset2.jpg')].features.t()
    ) > .7).sum().item() > 0
    assert torch.any(torch.matmul(
        output_by_file[os.path.join(base_path, 'berset1.jpg')].features,
        output_by_file[os.path.join(base_path, 'berset3.jpg')].features.t()
    ) > .7).sum().item() > 0
    assert torch.any(torch.matmul(
        output_by_file[os.path.join(base_path, 'berset2.jpg')].features,
        output_by_file[os.path.join(base_path, 'berset3.jpg')].features.t()
    ) > .7).sum().item() > 0
