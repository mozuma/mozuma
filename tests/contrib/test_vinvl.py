import os

from mlmodule.contrib.vinvl import VinVLDetector
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_vinvl_object_detection(torch_device):
    vinvl = VinVLDetector(device=torch_device, score_threshold=0.5)

    # Pretrained model
    vinvl.load()

    # Getting data
    base_path = os.path.join("tests", "fixtures", "objects")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    # Get labels and attributes
    labels = vinvl.get_labels()
    attribute_labels = vinvl.get_attribute_labels()

    # Getting features
    img_paths, detections = vinvl.bulk_inference(dataset, data_loader_options={'batch_size': 10})

    assert os.path.basename(img_paths[0]) == 'icrc_vehicle.jpg'
    assert os.path.basename(img_paths[1]) == 'soldiers.jpg'
    assert len(detections) == len(file_names)
    assert len(detections[0]) == 10
    assert labels[detections[0][0].labels] == 'sign'
    assert attribute_labels[detections[0][0].attributes[0]] == 'black'
    assert len(detections[1]) == 19
    assert labels[detections[1][0].labels] == 'gun'
    assert attribute_labels[detections[1][4].attributes[0]] == 'blue'
