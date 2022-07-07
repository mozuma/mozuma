---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3.8.13 (conda)
    language: python
    name: python3
---

# Train an age detection model based on ArcFace features


Import `mozuma` modules

```python
from mozuma.models.arcface.pretrained import torch_arcface_insightface
from mozuma.models.mtcnn.pretrained import torch_mtcnn
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.models.classification import LinearClassifierTorchModule
from mozuma.torch.datasets import TorchTrainingDataset
from mozuma.torch.runners import TorchTrainingRunner
from mozuma.torch.options import TorchTrainingOptions
from mozuma.torch.options import TorchRunnerOptions
from mozuma.labels.base import LabelSet
from mozuma.callbacks.memory import (
    CollectBoundingBoxesInMemory,
    CollectFeaturesInMemory,
    CollectLabelsInMemory,
)
from mozuma.torch.datasets import (
    ImageBoundingBoxDataset,
    ListDataset,
    LocalBinaryFilesDataset,
    ImageDataset,
)
from mozuma.helpers.files import list_files_in_dir

from ignite.metrics import Precision, Recall, Loss, Accuracy
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

import os
import re
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

First download `UTKFace_inthewild` dataset from https://drive.google.com/drive/folders/1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G
Download and extract the 3 parts and place the folder `UTKFace_inthewild` in your home directory.
part1 will serve as training set, while part2 will be our test set and part3 our valid set.

```python
path_to_utkface = os.path.join(os.environ["HOME"], "UTKFace_inthewild")
train_filenames = list_files_in_dir(
    os.path.join(path_to_utkface, "part1"), allowed_extensions=("jpg",)
)
test_filenames = list_files_in_dir(
    os.path.join(path_to_utkface, "part2"), allowed_extensions=("jpg",)
)
valid_filenames = list_files_in_dir(
    os.path.join(path_to_utkface, "part3"), allowed_extensions=("jpg",)
)
train_dataset = ImageDataset(LocalBinaryFilesDataset(train_filenames))
test_dataset = ImageDataset(LocalBinaryFilesDataset(test_filenames))
valid_dataset = ImageDataset(LocalBinaryFilesDataset(valid_filenames))
```

Extract pretrained ArcFace features for all images

```python
torch_device = "cuda"
face_detector = torch_mtcnn(device=torch_device)
face_extractor = torch_arcface_insightface(device=torch_device)


def get_face_features(dset):
    # Callbacks
    bb = CollectBoundingBoxesInMemory()
    ff = CollectFeaturesInMemory()

    # Face detection runner
    runner = TorchInferenceRunner(
        model=face_detector,
        dataset=dset,
        callbacks=[bb],
        options=TorchRunnerOptions(
            data_loader_options={"batch_size": 1},
            device=torch_device,
            tqdm_enabled=True,
        ),
    )
    runner.run()

    # Dataset with detected faces
    dataset = ImageBoundingBoxDataset(
        image_dataset=ImageDataset(LocalBinaryFilesDataset(bb.indices)),
        bounding_boxes=bb.bounding_boxes,
    )

    # Face extraction runner
    runner = TorchInferenceRunner(
        model=face_extractor,
        dataset=dataset,
        callbacks=[ff],
        options=TorchRunnerOptions(
            data_loader_options={"batch_size": 32},
            device=torch_device,
            tqdm_enabled=True,
        ),
    )
    runner.run()

    return bb, ff
```

```python
train_bb, train_ff = get_face_features(train_dataset)
test_bb, test_ff = get_face_features(test_dataset)
valid_bb, valid_ff = get_face_features(valid_dataset)
```

Discretising the age into six categories

```python
childhood = {i: "childhood" for i in range(0, 8)}
puberty = {i: "puberty" for i in range(8, 13)}
adolescence = {i: "adolescence" for i in range(13, 18)}
adulthood = {i: "adulthood" for i in range(18, 35)}
middle_age = {i: "middle_age" for i in range(35, 50)}
seniority = {i: "seniority" for i in range(50, 120)}
age2label = {
    **childhood,
    **puberty,
    **adolescence,
    **adulthood,
    **middle_age,
    **seniority,
}


def discretize_age(filenames):
    labels = {}
    for img_path in filenames:
        m = re.search("(\d+)_.*[.jpg]", img_path)
        if m:
            age = int(m.group(1))
            labels[img_path] = age2label[age]
        else:
            print(f"{img_path} failed")
    assert len(labels) == len(filenames)
    return labels
```

```python
train_labels = discretize_age(train_filenames)
test_labels = discretize_age(test_filenames)
valid_labels = discretize_age(valid_filenames)

label_set = LabelSet(
    label_set_unique_id="age",
    label_list=[
        "childhood",
        "puberty",
        "adolescence",
        "adulthood",
        "middle_age",
        "seniority",
    ],
)
```

Define train and validation set for training the classifier

```python
# define train set
train_dset = TorchTrainingDataset(
    dataset=ListDataset(train_ff.features),
    targets=label_set.get_label_ids(
        [train_labels[img_path] for img_path, _ in train_ff.indices]
    ),
)
# define valid set
valid_dset = TorchTrainingDataset(
    dataset=ListDataset(valid_ff.features),
    targets=label_set.get_label_ids(
        [valid_labels[img_path] for img_path, _ in valid_ff.indices]
    ),
)
```

Define a linear classifier on top of the extracted features

```python
age_classifier = LinearClassifierTorchModule(
    in_features=train_ff.features.shape[1], label_set=label_set
)
```

Training runner

```python
precision = Precision(average=False)
recall = Recall(average=False)
F1 = (precision * recall * 2 / (precision + recall)).mean()

loss_fn = F.cross_entropy
trainer = TorchTrainingRunner(
    model=age_classifier,
    dataset=(train_dset, valid_dset),
    callbacks=[],
    options=TorchTrainingOptions(
        data_loader_options={"batch_size": 32},
        criterion=loss_fn,
        optimizer=optim.Adam(age_classifier.parameters(), lr=1e-3),
        metrics={
            "pre": precision,
            "recall": recall,
            "f1": F1,
            "acc": Accuracy(),
            "ce_loss": Loss(loss_fn),
        },
        validate_every=1,
        num_epoch=3,
        tqdm_enabled=True,
    ),
)
trainer.run()
```

Get predictions on the test set

```python
# Callbacks
score_test = CollectLabelsInMemory()

# Do the predictions
runner = TorchInferenceRunner(
    model=age_classifier,
    dataset=ListDataset(test_ff.features),
    callbacks=[score_test],
    options=TorchRunnerOptions(
        data_loader_options={"batch_size": 32}, device=torch_device, tqdm_enabled=True
    ),
)
runner.run()
```

Print the classification report

```python
test_ff_labels = [test_labels[img_path] for img_path, _ in test_ff.indices]
print(classification_report(test_ff_labels, score_test.labels))
```
