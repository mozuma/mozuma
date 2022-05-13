---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.7.10 ('mlmodule')
    language: python
    name: python3
---

# CIFAR10 Image Classification training

In this notebook, we are training CIFAR10 image classification on top of ResNet18 features from ImageNet


Import `mlmodule` modules for the task

```python
from mlmodule.models.resnet.modules import TorchResNetModule
from mlmodule.models.classification import LinearClassifierTorchModule
from mlmodule.torch.datasets import TorchTrainingDataset
from mlmodule.torch.runners import TorchTrainingRunner
from mlmodule.torch.runners import TorchInferenceRunner
from mlmodule.torch.options import TorchTrainingOptions
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.labels.base import LabelSet
from mlmodule.callbacks.memory import (
    CollectFeaturesInMemory,
)
from mlmodule.torch.datasets import (
    ListDataset,
    ListDatasetIndexed,
)
from mlmodule.states import StateKey
from mlmodule.stores import Store
```

Enable logging into notebook

```python
import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
```

Load CIFAR10 dataset from torchvision

```python
import os
from torchvision.datasets import CIFAR10

root_dir = os.path.join(os.environ["HOME"], 'torchvision-datasets')
train_cifar10 = CIFAR10(root=root_dir, train=True, download=True,  transform=None)
```

Format inputs and labels for `mlmodule`

```python
labels_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
train_samples = [(img, labels_dict[label]) for img, label in train_cifar10]
train_images, train_labels = zip(*train_samples)
```

Load `resnet18` pre-trained model

```python
torch_device = "cuda"
resnet = TorchResNetModule(
    resnet_arch="resnet18",
    device=torch_device,
    training_mode="features"
)
Store().load(resnet, StateKey(resnet.state_type, training_id="imagenet"))

```

Extract image features

```python
# Callbacks
ff = CollectFeaturesInMemory()

# Runner
runner = TorchInferenceRunner(
    model=resnet,
    dataset=ListDataset(train_images),
    callbacks=[ff],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
runner.run()
```

Create a linear classifier on top of ResNet features

```python
from mlmodule.models.classification import LinearClassifierTorchModule

labels = list(labels_dict.values())
labels.sort()
label_set = LabelSet(
            label_set_unique_id="cifar10",
            label_list=labels
        )

classifier = LinearClassifierTorchModule(
    in_features=ff.features.shape[1],
    label_set=label_set
)
```

Create train and validation splits

```python
import torch

# split samples into train and valid sets
train_indices, valid_indices = torch.split(torch.randperm(len(ff.indices)), int(len(ff.indices)*.9))
# define training set
train_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(train_indices, ff.features[train_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in train_indices])
)
# define valid set
valid_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(valid_indices, ff.features[valid_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in valid_indices])
)
```

Train the image classifier using `TorchTrainingRunner` module

```python
from ignite.metrics import Precision, Recall, Loss, Accuracy
from mlmodule.callbacks.states import SaveModelState
from mlmodule.stores.local import LocalStateStore

import torch.nn.functional as F
import torch.optim as optim

# define the evaluation metrics
precision = Precision(average=False)
recall = Recall(average=False)
F1 = (precision * recall * 2 / (precision + recall)).mean()

# Callbacks
model_state = SaveModelState(
    store=LocalStateStore('/home/lebret/data/mlmodule'),
    state_key=StateKey(classifier.state_type, 'train-1'))
# define a loss function
loss_fn =  F.cross_entropy

# define the trainer
trainer = TorchTrainingRunner(
    model=classifier,
    dataset=(train_dset, valid_dset),
    callbacks=[model_state],
    options=TorchTrainingOptions(
        data_loader_options={'batch_size': 32},
        criterion=loss_fn,
        optimizer=optim.Adam(classifier.parameters(), lr=1e-3),
        metrics={
            "pre": precision,
            "recall": recall,
            "f1": F1,
            "acc": Accuracy(),
            "ce_loss": Loss(loss_fn),
        },
        validate_every=1,
        checkpoint_every=3,
        num_epoch=5,
        tqdm_enabled=True,
    ),
)
trainer.run()
```

Do evaluation on the test set

```python
from mlmodule.callbacks.memory import CollectLabelsInMemory

test_cifar10 = CIFAR10(root=root_dir, train=False, download=True,  transform=None)
test_samples = [(img, labels_dict[label]) for img, label in test_cifar10]
test_images, test_labels = zip(*test_samples)

# Callbacks
ff_test = CollectFeaturesInMemory()
score_test = CollectLabelsInMemory()

# Extract the image features
features_test_runner = TorchInferenceRunner(
    model=resnet,
    dataset=ListDataset(test_images),
    callbacks=[ff_test],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
features_test_runner.run()

# Do the predictions
scores_test_runner = TorchInferenceRunner(
    model=classifier,
    dataset=ListDataset(ff_test.features),
    callbacks=[score_test],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
scores_test_runner.run()
```

Print classification report

```python
from sklearn.metrics import classification_report
print(classification_report(test_labels, score_test.labels))

```
