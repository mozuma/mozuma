import os
import torch

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_mtcnn_detector_inference(device=torch.device('cpu')):
    mtcnn = MTCNNDetector(device=device, image_size=720, min_face_size=20)
    # Pretrained model
    mtcnn.load()
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    outputs = mtcnn.bulk_inference(dataset)
    assert len(outputs) == 5
    assert outputs[0].boxes.shape[0] == 12


def test_mtcnn_detector_inference_gpu():
    if torch.cuda.is_available():
        test_mtcnn_detector_inference(torch.device('cuda:0'))
