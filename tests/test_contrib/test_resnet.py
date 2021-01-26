import os

from mlmodule.contrib.resnet import ResNet18ImageNetFeatures
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_resnet_features_inference():
    resnet = ResNet18ImageNetFeatures()
    # Pretrained model
    resnet.load()
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    features = resnet.bulk_inference(dataset, batch_size=10)
    assert len(features) == 50
    assert len(features[0]) == 512
