# MTCNN

We are using `facenet-pytorch` to load pretrained MTCNN model, see https://github.com/timesler/facenet-pytorch.

# Face Detection

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

detected_faces = mtcnn.bulk_inference(data)
```
