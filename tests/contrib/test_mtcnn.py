import os

from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_mtcnn_detector_inference(torch_device):
    mtcnn = MTCNNDetector(device=torch_device, image_size=720, min_face_size=20)
    # Pretrained model
    mtcnn.load()
    base_path = os.path.join("tests", "fixtures", "faces")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
    dataset = ImageDataset(file_names)

    file_names, outputs = mtcnn.bulk_inference(dataset)
    output_by_file = dict(zip(file_names, outputs))
    assert len(outputs) == 5
    # It should be a namedtuple of len 3
    assert len(outputs[0]) == 3
    assert output_by_file[os.path.join(base_path, 'office2.jpg')].boxes.shape[0] == 4
