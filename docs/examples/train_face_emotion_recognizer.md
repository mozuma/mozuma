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

# Face emotion recognition training

This notebooks shows how to train a face emotion recognition model on top of ArcFace face features


Import MTCNN and ArcFace modules from `mlmodule`


```python
from mlmodule.models.arcface.pretrained import torch_arcface_insightface
from mlmodule.models.mtcnn.pretrained import torch_mtcnn
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.runners import TorchInferenceRunner
from mlmodule.callbacks.memory import (
    CollectBoundingBoxesInMemory,
    CollectFeaturesInMemory,
)
from mlmodule.torch.datasets import (
    ImageBoundingBoxDataset,
    ListDataset,
    ListDatasetIndexed,
)

from torchvision.datasets import FER2013

import os
```


Enable logging inside notebook

```python
import logging
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
```

Load a dataset containing images of faces annotated with emotion labels

We should first download [FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset from Kaggle and unzip `train.csv` and `test.csv` files.


```python
path_to_fer2013 = os.path.join(os.environ["HOME"], "torchvision-datasets")
train_set = FER2013(root=path_to_fer2013, split="train")
```

```python
# Training images
labels_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
train_samples = [(img.convert("RGB"), labels_dict[label]) for img, label in train_set]
train_images, train_labels = zip(*train_samples)
```

Run face detection on the images with `TorchMTCNNModule`


```python
torch_device = "cuda"
model = torch_mtcnn(device=torch_device)

# Callbacks
bb = CollectBoundingBoxesInMemory()

# Runner
runner = TorchInferenceRunner(
    model=model,
    dataset=ListDataset(train_images),
    callbacks=[bb],
    options=TorchRunnerOptions(
        data_loader_options={"batch_size": 32}, device=torch_device, tqdm_enabled=True
    ),
)
runner.run()
```


Extract face features with `TorchArcFaceModule`


```python
arcface = torch_arcface_insightface(device=torch_device)

# Dataset
dataset = ImageBoundingBoxDataset(
    image_dataset=ListDatasetIndexed(indices=bb.indices, objects=train_images),
    bounding_boxes=bb.bounding_boxes,
)

# Callbacks
ff = CollectFeaturesInMemory()

# Runner
runner = TorchInferenceRunner(
    model=arcface,
    dataset=dataset,
    callbacks=[ff],
    options=TorchRunnerOptions(
        data_loader_options={"batch_size": 32}, device=torch_device, tqdm_enabled=True
    ),
)
runner.run()
```

Training of a linear classifier on top of the face features


Import the module for training

```python
from mlmodule.models.classification import LinearClassifierTorchModule
from mlmodule.torch.datasets import TorchTrainingDataset
from mlmodule.torch.runners import TorchTrainingRunner
from mlmodule.torch.options import TorchTrainingOptions
from mlmodule.labels.base import LabelSet

import torch
import torch.nn.functional as F
import torch.optim as optim
```

Define the training dataset


Define the labels

```python
labels = list(labels_dict.values())
label_set = LabelSet(label_set_unique_id="emotion", label_list=labels)
```

```python
# split samples into train and valid sets
train_indices, valid_indices = torch.split(
    torch.randperm(len(ff.indices)), int(len(ff.indices) * 0.9)
)
# define training set
train_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(train_indices, ff.features[train_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in train_indices]),
)
# define valid set
valid_dset = TorchTrainingDataset(
    dataset=ListDatasetIndexed(valid_indices, ff.features[valid_indices]),
    targets=label_set.get_label_ids([train_labels[idx] for idx in valid_indices]),
)
```

Define the linear classifier

```python
in_features = len(ff.features[0])

classifier = LinearClassifierTorchModule(in_features=in_features, label_set=label_set)
```

Define the trainer

```python
from ignite.metrics import Precision, Recall, Loss, Accuracy

precision = Precision(average=False)
recall = Recall(average=False)
F1 = (precision * recall * 2 / (precision + recall)).mean()

loss_fn = F.cross_entropy
trainer = TorchTrainingRunner(
    model=classifier,
    dataset=(train_dset, valid_dset),
    callbacks=[],
    options=TorchTrainingOptions(
        data_loader_options={"batch_size": 32},
        criterion=loss_fn,
        optimizer=optim.Adam(classifier.parameters(), lr=1e-2),
        metrics={
            "pre": precision,
            "recall": recall,
            "f1": F1,
            "acc": Accuracy(),
            "ce_loss": Loss(loss_fn),
        },
        validate_every=1,
        num_epoch=5,
        tqdm_enabled=True,
    ),
)
trainer.run()
```
