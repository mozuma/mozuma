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
from mlmodule.models.classification import LinearClassifierTorchModule
from mlmodule.models.resnet.pretrained import torch_resnet_imagenet
from mlmodule.models.densenet.pretrained import torch_densenet_imagenet
from mlmodule.torch.runners import TorchTrainingRunner
from mlmodule.torch.runners import TorchInferenceRunner
from mlmodule.torch.runners import TorchInferenceMultiGPURunner
from mlmodule.torch.options import TorchTrainingOptions
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.options import TorchMultiGPURunnerOptions
from mlmodule.labels.base import LabelSet
from mlmodule.torch.datasets import (
    ListDataset,
    ListDatasetIndexed,
    TorchTrainingDataset
)
from mlmodule.callbacks.memory import (
    CollectFeaturesInMemory,
    CollectLabelsInMemory
)
from mlmodule.callbacks.states import SaveModelState
from mlmodule.stores.local import LocalStateStore
from mlmodule.states import StateKey

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from ignite.metrics import Precision, Recall, Loss, Accuracy
from sklearn.metrics import classification_report
import os
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
resnet = torch_resnet_imagenet(
    resnet_arch="resnet18",
    device=torch_device,
    training_mode="features"
)

```

Extract image features from ResNet with a single GPU

```python
# Callbacks
ff_train_resnet = CollectFeaturesInMemory()

# Runner
runner = TorchInferenceRunner(
    model=resnet,
    dataset=ListDataset(train_images),
    callbacks=[ff_train_resnet],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
runner.run()
```

Create train and validation splits

```python
# define the set of labels
label_set = LabelSet(
            label_set_unique_id="cifar10",
            label_list=list(labels_dict.values())
        )

# split samples into train and valid sets
train_indices, valid_indices = torch.split(torch.randperm(len(ff_train_resnet.indices)), int(len(ff_train_resnet.indices)*.9))
# define training set
train_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(train_indices, ff_train_resnet.features[train_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in train_indices])
)
# define valid set
valid_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(valid_indices, ff_train_resnet.features[valid_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in valid_indices])
)
```

Train the image classifier using `TorchTrainingRunner` module

```python
# define a classifier on top of resnet features
classifier_resnet = LinearClassifierTorchModule(
    in_features=ff_train_resnet.features.shape[1],
    label_set=label_set
)

# define the evaluation metrics
precision = Precision(average=False)
recall = Recall(average=False)
F1 = (precision * recall * 2 / (precision + recall)).mean()

# Callbacks
exp_dir = os.path.join(os.environ["HOME"], 'mlmodule-training')
os.makedirs(exp_dir, exist_ok=True )

resnet_state = SaveModelState(
    store=LocalStateStore(exp_dir),
    state_key=StateKey(classifier_resnet.state_type, 'train-resnet'))

# define a loss function
loss_fn =  F.cross_entropy

# define the trainer
trainer = TorchTrainingRunner(
    model=classifier_resnet,
    dataset=(train_dset, valid_dset),
    callbacks=[resnet_state],
    options=TorchTrainingOptions(
        data_loader_options={'batch_size': 32},
        criterion=loss_fn,
        optimizer=optim.Adam(classifier_resnet.parameters(), lr=1e-3),
        metrics={
            "pre": precision,
            "recall": recall,
            "f1": F1,
            "acc": Accuracy(),
            "ce_loss": Loss(loss_fn),
        },
        validate_every=1,
        checkpoint_every=3,
        num_epoch=3,
        tqdm_enabled=True,
    ),
)
trainer.run()
```

Do evaluation on the test set

```python
test_cifar10 = CIFAR10(root=root_dir, train=False, download=True,  transform=None)
test_samples = [(img, labels_dict[label]) for img, label in test_cifar10]
test_images, test_labels = zip(*test_samples)

# Callbacks
ff_test_resnet = CollectFeaturesInMemory()
score_test_resnet = CollectLabelsInMemory()

# Extract the image features
runner = TorchInferenceRunner(
    model=resnet,
    dataset=ListDataset(test_images),
    callbacks=[ff_test_resnet],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
runner.run()

# Do the predictions
runner = TorchInferenceRunner(
    model=classifier_resnet,
    dataset=ListDataset(ff_test_resnet.features),
    callbacks=[score_test_resnet],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 32},
        device=torch_device,
        tqdm_enabled=True
    ),
)
runner.run()
```

Print classification report

```python
print(classification_report(test_labels, score_test_resnet.labels))
```

Compare the classification performance with a deeper DenseNet model


Load `densenet201` model

```python
torch_device = "cuda"
densenet = torch_densenet_imagenet(
    densenet_arch="densenet201",
    device=torch_device,
)
```

Extract image features with multiple gpus

```python
# Callbacks
ff_train_densenet = CollectFeaturesInMemory()

# Runner
runner = TorchInferenceMultiGPURunner(
    model=densenet,
    dataset=ListDataset(train_images),
    callbacks=[ff_train_densenet],
    options=TorchMultiGPURunnerOptions(
        data_loader_options={'batch_size': 32},
        tqdm_enabled=True
    ),
)
runner.run()
```

Train a new classifier on top of densenet201 features

```python
# define training set
train_dset_densenet = TorchTrainingDataset(
    dataset=ListDatasetIndexed(train_indices, ff_train_densenet.features[train_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in train_indices])
)
# define valid set
valid_dset_densenet = TorchTrainingDataset(
    dataset=ListDatasetIndexed(valid_indices, ff_train_densenet.features[valid_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in valid_indices])
)
```

```python
# define a classifier on top of resnet features
classifier_densenet = LinearClassifierTorchModule(
    in_features=ff_train_densenet.features.shape[1],
    label_set=label_set
)
# save state of the classifier on top of densenet features
densenet_state = SaveModelState(
    store=LocalStateStore(exp_dir),
    state_key=StateKey(classifier_densenet.state_type, 'train-densenet')
)

# define the trainer
trainer = TorchTrainingRunner(
    model=classifier_densenet,
    dataset=(train_dset_densenet, valid_dset_densenet),
    callbacks=[densenet_state],
    options=TorchTrainingOptions(
        data_loader_options={'batch_size': 32},
        criterion=loss_fn,
        optimizer=optim.Adam(classifier_densenet.parameters(), lr=1e-3),
        metrics={
            "pre": precision,
            "recall": recall,
            "f1": F1,
            "acc": Accuracy(),
            "ce_loss": Loss(loss_fn),
        },
        validate_every=1,
        checkpoint_every=3,
        num_epoch=3,
        tqdm_enabled=True,
    ),
)
trainer.run()
```

```python
# Callbacks
ff_test_densenet = CollectFeaturesInMemory()
score_test_densenet = CollectLabelsInMemory()

# Extract the image features
runner = TorchInferenceMultiGPURunner(
    model=densenet,
    dataset=ListDataset(test_images),
    callbacks=[ff_test_densenet],
    options=TorchMultiGPURunnerOptions(
        data_loader_options={'batch_size': 32},
        tqdm_enabled=True
    ),
)
runner.run()

# Do the predictions
runner = TorchInferenceMultiGPURunner(
    model=classifier_densenet,
    dataset=ListDataset(ff_test_densenet.features),
    callbacks=[score_test_densenet],
    options=TorchMultiGPURunnerOptions(
        data_loader_options={'batch_size': 32},
        tqdm_enabled=True
    ),
)
runner.run()

print(classification_report(test_labels, score_test_densenet.labels))
```

From the classification report we can see that performance with densenet201 are much better than the performance with resnet18. 
