import os

from mlmodule.contrib.densenet import DenseNet161ImageNetFeatures, DenseNet161PlacesFeatures, \
    DenseNet161ImageNetClassifier, DenseNet161PlacesClassifier
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


def test_densenet_imagenet_features_inference(torch_device):
    densenet = DenseNet161ImageNetFeatures(device=torch_device)

    # Pretrained model
    densenet.load()
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    file_names_idx, features = densenet.bulk_inference(dataset, batch_size=10)
    assert len(features) == 50
    assert len(features[0]) == 2208
    assert type(file_names[0]) == str
    assert set(file_names_idx) == set(file_names)


def test_densenet_places365_features_inference(torch_device):
    densenet = DenseNet161PlacesFeatures(device=torch_device)

    # Pretrained model
    densenet.load()
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    file_names_idx, features = densenet.bulk_inference(dataset, batch_size=10)
    assert len(features) == 50
    assert len(features[0]) == 2208
    assert type(file_names[0]) == str
    assert set(file_names_idx) == set(file_names)


def test_densenet_imagenet_classifier(torch_device):
    densenet = DenseNet161ImageNetFeatures(device=torch_device)
        
    # Pretrained model
    densenet.load()

    # Getting data
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    # Getting features
    idx, features = densenet.bulk_inference(dataset, batch_size=10)

    # Creating features dataset
    features = IndexedDataset(idx, features)   # Zipping indices and features

    # Getting classifier
    densenet_cls = DenseNet161ImageNetClassifier()
    densenet_cls.load()

    # Running inference
    file_names_idx, weights = densenet_cls.bulk_inference(features)
    max_class = weights.argmax(axis=1)
    # Putting class label
    label_set = densenet_cls.get_labels()
    max_class = [label_set[c] for c in max_class]

    # Collecting classes with filenames
    file_class = dict(zip(file_names_idx, max_class))

    # Making sure we have all input classified
    assert set(file_names) == set(file_names_idx)

    # Verifying a couple of output labels
    assert 'cat' in file_class[os.path.join(base_path, "cat_90.jpg")].lower()
    assert file_class[os.path.join(base_path, "dog_900.jpg")] == 'Labrador retriever'


def test_densenet_places365_classifier(torch_device):
    densenet = DenseNet161PlacesFeatures(device=torch_device)
        
    # Pretrained model
    densenet.load()

    # Getting data
    base_path = os.path.join("tests", "fixtures", "cats_dogs")
    file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
    dataset = ImageDataset(file_names)

    # Getting features
    idx, features = densenet.bulk_inference(dataset, batch_size=10)

    # Creating features dataset
    features = IndexedDataset(idx, features)   # Zipping indices and features

    # Getting classifier
    densenet_cls = DenseNet161PlacesClassifier()
    densenet_cls.load()

    # Running inference
    file_names_idx, weights = densenet_cls.bulk_inference(features)
    max_class = weights.argmax(axis=1)
    # Putting class label
    label_set = densenet_cls.get_labels()
    max_class = [label_set[c] for c in max_class]

    # Collecting classes with filenames
    file_class = dict(zip(file_names_idx, max_class))

    # Making sure we have all input classified
    assert set(file_names) == set(file_names_idx)

    # Verifying a couple of output labels
    assert 'veterinarians' in file_class[os.path.join(base_path, "cat_90.jpg")].lower()
    assert 'veterinarians' in file_class[os.path.join(base_path, "dog_900.jpg")].lower()
