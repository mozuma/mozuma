# MTCNN

!!! warning "Deprecated"
    This model needs to be migrated to the latest version

We are using `facenet-pytorch` to load pretrained MTCNN model, see `https://github.com/timesler/facenet-pytorch`.

## Requirements

This needs mlmodule with the mtcnn and torch extra requirements:

```bash
pip install git+ssh://git@github.com/LSIR/mlmodule.git#egg=mlmodule[torch,mtcnn]
# or
pip install mlmodule[torch,mtcnn]
```


## Face Detection

Usage:

```python
import torch
from mlmodule.contrib.mtcnn import MTCNNDetector
from mlmodule.torch.data.images import ImageDataset
from mlmodule.utils import list_files_in_dir

# We need a list of files
file_list = list_files_in_dir('mlmodule/tests/fixtures/faces', allowed_extensions='jpg')

mtcnn = MTCNNDetector(device=torch.device('cpu')).load()
data = ImageDataset(file_list)

file_list, detected_faces = mtcnn.bulk_inference(data)
```

## Download the model weights on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download mtcnn.MTCNNDetector "$PUBLIC_ASSETS/face-detection/mtcnn.pt"
```
