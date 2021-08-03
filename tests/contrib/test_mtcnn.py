import os
from typing import Tuple, List

import numpy as np
import pytest
from facenet_pytorch.models.mtcnn import MTCNN
from torchvision.transforms import Compose, Resize

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.box import BBoxOutput, BBoxPoint
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import convert_to_rgb, get_pil_image_from_file, LoadRGBPILImage
from mlmodule.utils import list_files_in_dir


@pytest.fixture(scope='session')
def mtcnn_instance(torch_device):
    return MTCNNDetector(device=torch_device, min_face_size=20)


@pytest.fixture(scope='session')
def resized_images() -> Tuple[List[str], List[np.ndarray]]:
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    transforms = Compose([
        LoadRGBPILImage(shrink_image_size=(1440, 1440)),
        Resize((1440, 1440))
    ])
    return file_names, [transforms(f) for f in file_names]


@pytest.fixture(scope='session')
def mtcnn_inference_results(mtcnn_instance, resized_images):
    mtcnn = mtcnn_instance
    # Pretrained model
    mtcnn.load()
    indices, images = resized_images
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](indices, images)
    return mtcnn.bulk_inference(dataset)


def assert_bbox_equals(first: BBoxOutput, second: BBoxOutput):
    assert first.bounding_box == second.bounding_box
    assert first.probability == second.probability
    np.testing.assert_equal(first.features, second.features)


def test_mtcnn_detector_inference(mtcnn_inference_results):
    file_names, outputs = mtcnn_inference_results

    output_by_file = dict(zip(file_names, outputs))
    assert len(outputs) == 3
    # It should be a BBoxOutput
    assert type(outputs[0][0]) == BBoxOutput
    assert len(output_by_file[os.path.join("tests", "fixtures", "berset", 'berset2.jpg')]) == 8


def test_mtcnn_detector_inference_no_faces(mtcnn_instance):
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = sorted(list_files_in_dir(base_path, allowed_extensions=('jpg',)))[:5]
    transforms = Compose([
        get_pil_image_from_file,
        convert_to_rgb,
        Resize((720, 720))
    ])
    data = [transforms(f) for f in file_names]
    mtcnn = mtcnn_instance
    # Pretrained model
    mtcnn.load()
    dataset = IndexedDataset[str, np.ndarray, np.ndarray](file_names, data)
    file_names, bbox = mtcnn.bulk_inference(dataset)

    for b in bbox:
        assert len(b) == 0


def test_mtcnn_detector_correctness(mtcnn_inference_results, mtcnn_instance, torch_device, resized_images):
    file_names, outputs = mtcnn_inference_results
    mtcnn_orig = MTCNN(device=torch_device, min_face_size=20)

    # Testing first image
    f, images = resized_images
    assert f == file_names
    transforms = Compose(mtcnn_instance.get_dataset_transforms())
    all_boxes, all_probs, all_landmarks = mtcnn_orig.detect(
        images, landmarks=True
    )
    for f, bbox_col, (boxes, probs, landmarks) in zip(file_names, outputs, zip(all_boxes, all_probs, all_landmarks)):
        # Ordering bounding boxes for comparison
        for bbox, (box, prob, features) in zip(
                sorted(bbox_col, key=lambda b: b.bounding_box[0][0]),
                sorted(zip(boxes, probs, landmarks), key=lambda b: b[0][0])
        ):
            expected_bbox = BBoxOutput(
                bounding_box=(BBoxPoint(*box[:2]), BBoxPoint(*box[2:])),
                probability=prob,
                features=features
            )
            np.testing.assert_allclose(
                np.array(bbox.bounding_box), np.array(expected_bbox.bounding_box),
                rtol=0.5
            )
            np.testing.assert_allclose(
                np.array(bbox.features), np.array(expected_bbox.features),
                rtol=0.5
            )


def test_mtcnn_serialisation(mtcnn_inference_results, mtcnn_instance):
    file_names, outputs = mtcnn_inference_results

    # Checking we can recover the same data
    for bounding_boxes in outputs:
        for bbox in bounding_boxes:
            s_features = mtcnn_instance.from_binary(mtcnn_instance.to_binary(bbox.features))
            np.testing.assert_equal(bbox.features, s_features)
