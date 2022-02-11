# ResNet


TODO: Describe pretrained weight and initialisation parameters.

We are using torchvision to load pretrained models, see https://pytorch.org/docs/stable/torchvision/models.html.

# Feature models

Usage:

```python
from mlmodule.contrib.resnet import ResNet18ImageNetFeatures
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

# We need a list of files
file_list = list_files_in_dir('mlmodule/tests/fixtures/cats_dogs', allowed_extensions='jpg')

resnet = ResNet18ImageNetFeatures('resnet18').load()
data = ImageDataset(file_list)

file_names, features = resnet.bulk_inference(data)
```

# Pretrained classifier

Usage:

```python
from mlmodule.contrib.resnet import ResNet18ImageNetClassifier
from mlmodule.torch.data.base import IndexedDataset

resnet_classifier = ResNet18ImageNetClassifier('resnet18').load()
# We need features from the previous step
data = IndexedDataset(file_names, features)

file_names, pred = resnet_classifier.bulk_inference(data)
```
