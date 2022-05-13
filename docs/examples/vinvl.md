---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.7.11 64-bit (conda)
    name: python3
---

# Object Detection with VinVL


Import `mlmodule` modules

```python
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.torch.runners import TorchInferenceRunner
from mlmodule.callbacks.memory import (
    CollectBoundingBoxesInMemory,
)
from mlmodule.helpers.files import list_files_in_dir
from mlmodule.torch.datasets import (
    LocalBinaryFilesDataset,
    ImageDataset
)
from mlmodule.models.vinvl.pretrained import torch_vinvl_detector

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
%matplotlib inline
```

Load images

```python
base_path = os.path.join("../../tests", "fixtures", "objects")
file_names = list_files_in_dir(base_path, allowed_extensions=('jpg',))[:50]
dataset = ImageDataset(LocalBinaryFilesDataset(file_names))
```

Run object detection with `torch_vinvl_detector`

```python
# Load VinVL model (it might take a few minutes.)
torch_device = torch.device("cuda")
vinvl = torch_vinvl_detector(device=torch_device, score_threshold=0.5)

bb = CollectBoundingBoxesInMemory()

# Runner
runner = TorchInferenceRunner(
    model=vinvl,
    dataset=dataset,
    callbacks=[bb],
    options=TorchRunnerOptions(
        device=torch_device,
        data_loader_options={"batch_size": 10},
        tqdm_enabled=True
    ),
)
runner.run()
```

Visualise the detected objects


First get labels and attributes

```python
for i, img_path in enumerate(bb.indices):
    print(f'Object detected for {img_path}')
    img = Image.open(img_path).convert('RGB')
    plt.figure()
    plt.imshow(img)
    bboxes = bb.bounding_boxes[i].bounding_boxes
    scores = bb.bounding_boxes[i].scores
    for k, bbox in enumerate(bboxes):
        bbox0, bbox1, bbox2, bbox3 = bbox
        plt.gca().add_patch(Rectangle((bbox0, bbox1),
                                        bbox2 - bbox0,
                                        bbox3 - bbox1, fill=False,
                                      edgecolor='red', linewidth=2, alpha=0.5))
        plt.text(
            bbox0, bbox1, f'{scores[k]*100:.1f}%', color='blue', fontsize=12)

```
