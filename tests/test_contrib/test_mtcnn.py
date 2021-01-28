import os
import pytest
import torch

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_mtcnn_detector_inference(device=torch.device('cpu')):
    mtcnn = MTCNNDetector(device=device, image_size=720, min_face_size=20)
    # Pretrained model
    mtcnn.load()
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    dataset = ImageDataset(file_names)

    file_names, outputs = mtcnn.bulk_inference(dataset)
    assert len(outputs) == 5
    # It should be a namedtuple of len 3
    assert len(outputs[0]) == 3
    assert outputs[0].boxes.shape[0] == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_mtcnn_detector_inference_gpu():
    test_mtcnn_detector_inference(torch.device('cuda:0'))
