import os
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
from facenet_pytorch.models.mtcnn import MTCNN
from PIL.Image import Image

from mlmodule.models.mtcnn.modules import TorchMTCNNModule
from mlmodule.utils import list_files_in_dir
from mlmodule.v2.helpers.callbacks import CollectBoundingBoxesInMemory
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.torch.datasets import (
    ImageDataset,
    ListDataset,
    LocalBinaryFilesDataset,
)
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


@pytest.fixture(scope="session")
def mtcnn_instance(torch_device: torch.device) -> TorchMTCNNModule:
    return TorchMTCNNModule(device=torch_device)


@pytest.fixture(scope="session")
def resized_images() -> Tuple[List[str], List[Image]]:
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))
    dataset = ImageDataset(
        LocalBinaryFilesDataset(file_names), resize_image_size=((1440, 1440))
    )
    image_arr = [dataset[i][1] for i in range(len(dataset))]
    return file_names, image_arr


def _run_mtcnn_inference(
    image_paths: List[str],
    torch_device: torch.device,
    inference_device: Optional[torch.device] = None,
) -> CollectBoundingBoxesInMemory:
    mtcnn = TorchMTCNNModule(device=torch_device)
    # Pretrained model
    Store().load(
        mtcnn,
        state_key=StateKey(state_type=mtcnn.state_type, training_id="facenet"),
    )

    # Dataset
    dataset = ImageDataset(
        LocalBinaryFilesDataset(image_paths), resize_image_size=(1440, 1440)
    )

    # Callbacks
    result = CollectBoundingBoxesInMemory()

    # Runner
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=mtcnn,
        callbacks=[result],
        options=TorchRunnerOptions(device=inference_device or mtcnn.device),
    )
    runner.run()

    return result


def test_mtcnn_detector_inference(torch_device: torch.device):
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))
    results = _run_mtcnn_inference(file_names, torch_device)
    output_by_file = dict(zip(results.indices, results.bounding_boxes))
    assert len(results.bounding_boxes) == 3
    assert (
        len(
            output_by_file[
                os.path.join("tests", "fixtures", "berset", "berset2.jpg")
            ].bounding_boxes
        )
        == 8
    )


def test_mtcnn_detector_inference_no_faces(torch_device: torch.device):
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = sorted(list_files_in_dir(base_path, allowed_extensions=("jpg",)))[:2]
    results = _run_mtcnn_inference(file_names, torch_device)

    for f, b in zip(results.indices, results.bounding_boxes):
        assert (
            len(b.bounding_boxes) == 0
        ), f"Unexpected bounding box {b.bounding_boxes, b.scores} in {f}"


def test_mtcnn_detector_correctness(
    torch_device: torch.device, resized_images: Tuple[List[str], List[Image]]
):
    # MlModule implementation
    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))
    results = _run_mtcnn_inference(file_names, torch_device)

    # Original implementation
    mtcnn_orig = MTCNN(device=torch_device, min_face_size=20)

    # Testing first image
    f, images = resized_images
    assert f == results.indices

    all_boxes: List[np.ndarray]
    all_probs: List[np.ndarray]
    all_landmarks: List[np.ndarray]
    all_boxes, all_probs, all_landmarks = mtcnn_orig.detect(images, landmarks=True)
    for f, bbox_col, (boxes, probs, landmarks) in zip(
        file_names, results.bounding_boxes, zip(all_boxes, all_probs, all_landmarks)
    ):
        # Ordering bounding boxes for comparison
        sort_idx_mlm = np.argsort(bbox_col.bounding_boxes[:, 0])
        sort_idx_ori = np.argsort(boxes[:, 0])
        np.testing.assert_allclose(
            bbox_col.bounding_boxes[sort_idx_mlm, :],
            boxes[sort_idx_ori, :],
            rtol=0.5,
        )
        assert bbox_col.scores is not None
        np.testing.assert_allclose(
            bbox_col.scores[sort_idx_mlm],
            probs[sort_idx_ori],
            rtol=0.5,
        )
        assert bbox_col.features is not None
        np.testing.assert_allclose(
            bbox_col.features[sort_idx_mlm],
            landmarks[sort_idx_ori],
            rtol=0.5,
        )


def test_mtcnn_small_images(torch_device: torch.device):
    dataset = ListDataset(
        [np.random.randint(255, size=(14, 200, 3), dtype=np.uint8) for _ in range(2)]
    )

    mtcnn = TorchMTCNNModule(device=torch_device)
    # Pretrained model
    Store().load(
        mtcnn,
        state_key=StateKey(state_type=mtcnn.state_type, training_id="facenet"),
    )

    # Callbacks
    result = CollectBoundingBoxesInMemory()

    # Runner
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=mtcnn,
        callbacks=[result],
        options=TorchRunnerOptions(device=mtcnn.device),
    )
    runner.run()

    assert all(len(b.bounding_boxes) == 0 for b in result.bounding_boxes)


def test_mtcnn_to_device():
    """Tests that if initializing the model on CPU does not prevent it to run later on CUDA"""
    if not torch.cuda.is_available():
        pytest.skip("Need CUDA to run this test")

    base_path = os.path.join("tests", "fixtures", "berset")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))
    _run_mtcnn_inference(
        file_names, torch.device("cpu"), inference_device=torch.device("cuda")
    )
