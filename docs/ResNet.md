# ResNet

We are using torchvision to load pretrained models, see https://pytorch.org/docs/stable/torchvision/models.html.

# Feature models

Usage:

```python
from mlmodule.contrib.resnet import ResNetFeatures
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

# We need a list of files
file_list = list_files_in_dir('mlmodule/tests/fixtures/cats_dogs', allowed_extensions='jpg')

resnet = ResNetFeatures('resnet18').load()
data = ImageDataset(file_list)

features = resnet.bulk_inference(data)
```

# Pretrained classifier

Usage:

```python
from mlmodule.contrib.resnet import ResNetDefaultClassifier
from mlmodule.torch.data.base import BaseIndexedDataset


resnet_classifier = ResNetDefaultClassifier('resnet18').load()
# We need features from the previous step
data = BaseIndexedDataset(features)

pred = resnet_classifier.bulk_inference(data)
```

# Retrain classifier

TODO
