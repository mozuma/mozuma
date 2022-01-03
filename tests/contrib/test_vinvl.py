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
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))[:50]
    dataset = ImageDataset(file_names)

    # Get labels and attributes
    labels = vinvl.get_labels()
    attribute_labels = vinvl.get_attribute_labels()

    # Getting features
    img_paths, detections = vinvl.bulk_inference(
        dataset, data_loader_options={"batch_size": 10}
    )

    idx_icrc = next(
        i
        for i, path in enumerate(img_paths)
        if os.path.basename(path) == "icrc_vehicle.jpg"
    )
    idx_sol = next(
        i
        for i, path in enumerate(img_paths)
        if os.path.basename(path) == "soldiers.jpg"
    )

    assert len(detections) == len(file_names)
    assert len(detections[idx_icrc]) == 10
    assert labels[detections[idx_icrc][0].labels[0]] == "sign"
    assert attribute_labels[detections[idx_icrc][0].attributes[0]] == "black"
    assert len(detections[idx_sol]) == 19
    assert labels[detections[idx_sol][0].labels[0]] == "gun"
    assert attribute_labels[detections[idx_sol][4].attributes[0]] == "blue"
