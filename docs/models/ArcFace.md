# ArcFace

!!! warning "Deprecated"
    This model needs to be migrated to the latest version

We are using an implementation of InsightFace in PyTorch (https://github.com/TreB1eN/InsightFace_Pytorch)


## Usage


```python
import os
from typing import List

import torch

from mlmodule.box import BBoxOutput
from mlmodule.contrib.arcface import ArcFaceFeatures
from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.box import BoundingBoxDataset
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir


torch_device = torch.device('cpu')
base_path = os.path.join("tests", "fixtures", "berset")
file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))
# Load image dataset
dataset = ImageDataset(file_names)

# Getting face detection model
mtcnn = MTCNNDetector(device=torch_device, image_size=720,
                      min_face_size=20)
mtcnn.load()
# Detect faces first
file_names, outputs = mtcnn.bulk_inference(dataset)

# Flattening all detected faces
bboxes: List[BBoxOutput]
indices: List[str]
indices, file_names, bboxes = zip(*[
    (f'{fn}_{i}', fn, bbox) for fn, bbox_list in zip(file_names, outputs) for i, bbox in enumerate(bbox_list)
])

# Create a dataset for the bounding boxes
bbox_features = BoundingBoxDataset(indices, file_names, bboxes)

arcface = ArcFaceFeatures(device=torch_device)
arcface.load()
# Get face features
indices, new_outputs = arcface.bulk_inference(
    bbox_features, data_loader_options={'batch_size': 3})
```


## Download the model weights on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download arcface.ArcFaceFeatures --outdir "$PUBLIC_ASSETS/face-detection/"
```
