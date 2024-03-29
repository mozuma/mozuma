import os

from mozuma.callbacks.memory import CollectBoundingBoxesInMemory
from mozuma.helpers.files import list_files_in_dir
from mozuma.models.vinvl.pretrained import torch_vinvl_detector
from mozuma.torch.datasets import ImageDataset, LocalBinaryFilesDataset
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner


def test_vinvl_object_detection(torch_device):
    vinvl = torch_vinvl_detector(device=torch_device, score_threshold=0.5)

    # Getting data
    base_path = os.path.join("tests", "fixtures", "objects")
    file_names = list_files_in_dir(base_path, allowed_extensions=("jpg",))[:50]
    dataset = ImageDataset(LocalBinaryFilesDataset(file_names))

    # Get labels and attributes
    # labels = vinvl.get_labels()
    # attribute_labels = vinvl.get_attribute_labels()

    # Getting features
    bb = CollectBoundingBoxesInMemory()

    # Runner
    runner = TorchInferenceRunner(
        model=vinvl,
        dataset=dataset,
        callbacks=[bb],
        options=TorchRunnerOptions(
            device=torch_device, data_loader_options={"batch_size": 10}
        ),
    )
    runner.run()

    # filename of source images
    img_names = [os.path.basename(p) for p in bb.indices]

    # Index of icrc vehicle image
    idx_icrc = img_names.index("icrc_vehicle.jpg")
    idx_sol = img_names.index("soldiers.jpg")

    icrc_bbox = bb.bounding_boxes[idx_icrc]
    assert 8 < len(icrc_bbox.bounding_boxes) < 12
    assert icrc_bbox.bounding_boxes.shape[1] == 4
    assert icrc_bbox.scores is not None
    assert icrc_bbox.scores.shape == (len(icrc_bbox.bounding_boxes),)
    # assert labels[detections[idx_icrc][0].labels[0]] == "sign"
    # assert attribute_labels[detections[idx_icrc][0].attributes[0]] == "black"
    assert 17 < len(bb.bounding_boxes[idx_sol].bounding_boxes) < 22
    # assert labels[detections[idx_sol][0].labels[0]] == "gun"
    # assert attribute_labels[detections[idx_sol][4].attributes[0]] == "blue"
