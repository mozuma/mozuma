from conftest import device_parametrize
import os

from mlmodule.contrib.densenet import DenseNet161PlacesFeatures, DenseNet161PlacesClassifier
from mlmodule.contrib.places365 import PlacesIOClassifier
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


@device_parametrize
def test_places365_50_images(device):
    # Load Pretrained model
    densenet = DenseNet161PlacesFeatures(device=device)
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
    file_names_idx, predictions = densenet_cls.bulk_inference(features)
    # Exporting the predictions to numpy
    predictions = predictions.cpu().detach().numpy()

    # Loading the IO Classifier
    io_classifier = PlacesIOClassifier()
    # Evaluating for each image
    indoors = io_classifier.bulk_inference(predictions)

    # Collecting classes with filenames
    file_class = dict(zip(file_names_idx, indoors))

    # Making sure we have all input classified
    assert set(file_names) == set(file_names_idx)

    # Verifying a couple of output labels
    assert file_class[os.path.join(base_path, "cat_90.jpg")] == 1
    assert file_class[os.path.join(base_path, "cat_951.jpg")] == 1
    assert file_class[os.path.join(base_path, "cat_960.jpg")] == 1
    assert file_class[os.path.join(base_path, "dog_941.jpg")] == 1
    assert file_class[os.path.join(base_path, "dog_910.jpg")] == 0
    assert file_class[os.path.join(base_path, "dog_921.jpg")] == 0
    assert file_class[os.path.join(base_path, "dog_961.jpg")] == 0
    assert file_class[os.path.join(base_path, "dog_970.jpg")] == 0
