import os

from mlmodule.contrib.vinvl.modules import TorchVinVLDetectorModule
from mlmodule.utils import list_files_in_dir
from mlmodule.v2.helpers.callbacks import CollectBoundingBoxesInMemory
from mlmodule.v2.states import StateKey
from mlmodule.v2.stores import Store
from mlmodule.v2.torch.datasets import ImageDataset, LocalBinaryFilesDataset
from mlmodule.v2.torch.options import TorchRunnerOptions
from mlmodule.v2.torch.runners import TorchInferenceRunner


def test_vinvl_object_detection(torch_device):
    vinvl = TorchVinVLDetectorModule(device=torch_device, score_threshold=0.5)
    # Pretrained model
    Store().load(vinvl, StateKey(vinvl.state_type, training_id="vinvl"))

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

    assert 8 < len(bb.bounding_boxes[idx_icrc].bounding_boxes) < 12
    # assert labels[detections[idx_icrc][0].labels[0]] == "sign"
    # assert attribute_labels[detections[idx_icrc][0].attributes[0]] == "black"
    assert 17 < len(bb.bounding_boxes[idx_sol].bounding_boxes) < 22
    # assert labels[detections[idx_sol][0].labels[0]] == "gun"
    # assert attribute_labels[detections[idx_sol][4].attributes[0]] == "blue"
