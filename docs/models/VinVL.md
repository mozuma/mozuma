# VinVL: Revisiting Visual Representations in Vision-Language Models

Pre-trained large-scale object-attribute detection (OD) model is based on the ResNeXt-152 C4 architecture.
The OD model has been firstly trained on much larger amounts of data, combining multiple public object detection datasets, including [COCO](https://cocodataset.org/#home), [OpenImages (OI)](https://storage.googleapis.com/openimages/web/index.html), [Objects365](https://www.objects365.org/overview.html), and [Visual Genome (VG)](https://visualgenome.org/). Then it is fine-tuned on VG dataset alone, since VG is the only dataset with label attributes (see [issue #120](https://github.com/microsoft/Oscar/issues/120#issuecomment-898781183)). It predicts objects from 1594 classes with attributes from 524 classes.
See the [code](https://github.com/pzzhang/VinVL) and the [paper](https://arxiv.org/pdf/2101.00529.pdf) for details. 


## Requirements

The model needs the configuration system [YACS](https://github.com/rbgirshick/yacs) to be installed.

```bash
pip install yacs==0.1.8
```

## Usage
```python
from mlmodule.utils import list_files_in_dir
from mlmodule.torch.data.images import ImageDataset
from mlmodule.contrib.vinvl import VinVLDetector
from mlmodule.contrib.vinvl.utils import postprocess_attr
import torch
import os

# Load VinVL model
torch_device = torch.device('cuda')
vinvl = VinVLDetector(device=torch_device, score_threshold=0.5)
# Pretrained model
vinvl.load()

# Getting data
base_path = os.path.join("tests", "fixtures", "objects")
file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
dataset = ImageDataset(file_names)

# Get the detections
indices, detections = vinvl.bulk_inference(dataset, data_loader_options={'batch_size': 10})

# Get labels and attributes
labels = vinvl.get_labels()
attribute_labels = vinvl.get_attribute_labels()

# Print out detected object
for i, img_path in enumerate(indices):
    print(f'Object with attributes detected for {img_path}')
    for k, det in enumerate(detections[i]):
        label = labels[det.labels]
        attr_labels = det.attributes[det.attr_scores > 0.5]
        attr_scores = det.attr_scores[det.attr_scores > 0.5]
        attributes = postprocess_attr(attribute_labels, attr_labels, attr_scores)
        print(f'{k+1}: {",".join(list(attributes[0]))} {label} ({det.probability:.2f})')
```


## Download the model weights on PC32

```bash
export PUBLIC_ASSETS=/mnt/storage01/lsir-public-assets/pretrained-models
# For text encoders
python -m mlmodule.cli download vinvl.VinVLDetector --outdir "$PUBLIC_ASSETS/object-detection/"
```
