import os

from mlmodule.contrib.resnet import ResNet18ImageNetFeatures, ResNet18ImageNetClassifier
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_resnet_features_inference(torch_device):
    resnet = ResNet18ImageNetFeatures(device=torch_device)
    # Pretrained model
    resnet.load()
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    file_names_idx, features = resnet.bulk_inference(dataset, data_loader_options={'batch_size': 10})
    assert len(features) == 50
    assert len(features[0]) == 512
    assert type(file_names[0]) == str
    assert set(file_names_idx) == set(file_names)


def test_resnet_classifier(torch_device):
    resnet = ResNet18ImageNetFeatures(device=torch_device)
    # Pretrained model
    resnet.load()

    # Getting data
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    # Getting features
    idx, features = resnet.bulk_inference(dataset, data_loader_options={'batch_size': 10})

    # Creating features dataset
    features = IndexedDataset(idx, features)   # Zipping indices and features

    # Getting classifier
    resnet_cls = ResNet18ImageNetClassifier()
    resnet_cls.load()

    # Running inference
    file_names_idx, weights = resnet_cls.bulk_inference(features)
    max_class = weights.argmax(axis=1)
    # Putting class label
    label_set = resnet_cls.get_labels()
    max_class = [label_set[c] for c in max_class]

    # Collecting classes with filenames
    file_class = dict(zip(file_names_idx, max_class))

    # Making sure we have all input classified
    assert set(file_names) == set(file_names_idx)

    # Verifying a couple of output labels
    assert 'cat' in file_class[os.path.join(base_path, "cat_90.jpg")].lower()
    assert file_class[os.path.join(base_path, "dog_900.jpg")] == 'Labrador retriever'
