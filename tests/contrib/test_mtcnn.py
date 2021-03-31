import os

import numpy as np
import pytest
from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import Compose, Resize

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import convert_to_rgb, get_pil_image_from_file
from mlmodule.utils import list_files_in_dir


@pytest.fixture(scope='session')
def mtcnn_instance(torch_device):
    return MTCNNDetector(device=torch_device, image_size=720, min_face_size=20)


@pytest.fixture(scope='session')
def resized_images():
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    transforms = Compose([
        get_pil_image_from_file,
        convert_to_rgb,
        Resize((720, 720))
    ])
    return file_names, [transforms(f) for f in file_names]


@pytest.fixture(scope='session')
def mtcnn_inference_results(mtcnn_instance, resized_images, set_seeds):
    mtcnn = mtcnn_instance
    # Pretrained model
    mtcnn.load()
    dataset = IndexedDataset(*resized_images)
    set_seeds()
    return mtcnn.bulk_inference(dataset)


def test_mtcnn_detector_inference(mtcnn_inference_results):
    file_names, outputs = mtcnn_inference_results

    output_by_file = dict(zip(file_names, outputs))
    assert len(outputs) == 5
    # It should be a namedtuple of len 3
    assert len(outputs[0]) == 3
    assert output_by_file[os.path.join("tests", "fixtures", "faces", 'office2.jpg')].boxes.shape[0] == 4


def test_mtcnn_detector_correctness(mtcnn_inference_results, mtcnn_instance, torch_device, set_seeds, resized_images):
    file_names, outputs = mtcnn_inference_results
    mtcnn_orig = MTCNN(device=torch_device, min_face_size=20)

    # Testing first image
    set_seeds()
    _, images = resized_images
    transforms = Compose(mtcnn_instance.get_dataset_transforms())
    o_boxes, o_probs, o_landmarks = mtcnn_orig.detect(
        [transforms(i) for i in images], landmarks=True
    )
    for features, o_features in zip(outputs, zip(o_boxes, o_probs, o_landmarks)):
        for elem, o_elem in zip(features, o_features):
            assert elem.shape == o_elem.shape
            np.testing.assert_equal(elem, o_elem)


def test_mtcnn_serialisation(mtcnn_inference_results, mtcnn_instance):
    file_names, outputs = mtcnn_inference_results

    # Checking we can recover the same data
    for features in outputs:
        s_features = mtcnn_instance.from_binary(mtcnn_instance.to_binary(features))
        for elem, s_elem in zip(features, s_features):
            np.testing.assert_equal(elem, s_elem)
