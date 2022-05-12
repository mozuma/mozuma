---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3.9.12
    language: python
    name: python3
---

# Text-to-image retrieval with CLIP

This is an example of a text-to-Image retrieval engine based on OpenAI CLIP model


Import `mlmodule` modules for the task

```python
from mlmodule.torch.runners import TorchInferenceRunner
from mlmodule.torch.options import TorchRunnerOptions
from mlmodule.callbacks.memory import (
    CollectFeaturesInMemory,
)
from mlmodule.torch.datasets import (
    ImageDataset,
    ListDataset,
    LocalBinaryFilesDataset,
)
from mlmodule.helpers.files import list_files_in_dir

from mlmodule.models.clip.text import CLIPTextModule
from mlmodule.models.clip.image import CLIPImageModule

from mlmodule.states import StateKey
from mlmodule.stores import Store

import torch
```

Load CLIP Image Encoder

```python
image_encoder = CLIPImageModule(clip_model_name="ViT-B/32", device=torch.device("cuda"))
store = Store()
store.load(image_encoder, StateKey(image_encoder.state_type, "clip"))
```

Extract CLIP image features of FlickR30k dataset


It might take a few minutes for extracting the features...

```python
path_to_flickr30k_images = '/mnt/storage01/datasets/flickr30k/full/images'
file_names = list_files_in_dir(path_to_flickr30k_images, allowed_extensions=('jpg',))
dataset = ImageDataset(LocalBinaryFilesDataset(file_names))

image_features = CollectFeaturesInMemory()
runner = TorchInferenceRunner(
    dataset=dataset,
    model=image_encoder,
    callbacks=[image_features],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 128},
        device=image_encoder.device,
        tqdm_enabled=True
    ),
)
runner.run()
```

Load CLIP Text Encoder

```python
text_encoder = CLIPTextModule(image_encoder.clip_model_name, device=torch.device("cpu"))
store.load(text_encoder, StateKey(text_encoder.state_type, "clip"))
```

Extract CLIP text features of a given query

```python
text_queries = [
    "Workers look down from up above on a piece of equipment .",
    "Ballet dancers in a studio practice jumping with wonderful form ."
]
dataset = ListDataset(text_queries)

text_features = CollectFeaturesInMemory()
runner = TorchInferenceRunner(
    dataset=dataset,
    model=text_encoder,
    callbacks=[text_features],
    options=TorchRunnerOptions(
        data_loader_options={'batch_size': 1},
        device=text_encoder.device,
        tqdm_enabled=True
    ),
)
runner.run()
```

Text-to-image retrieval engine


Pick the top 5 most similar images for the text query


```python
img_feat = torch.tensor(image_features.features).type(torch.float32)
img_feat /= img_feat.norm(dim=-1, keepdim=True)
txt_feat = torch.tensor(text_features.features)
txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
similarity = (100.0 * txt_feat @ img_feat.T).softmax(dim=-1)
values, indices = similarity.topk(5)

```

Display the results

```python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install ipyplot
```

```python
import ipyplot
from PIL import Image
for k, text in enumerate(text_queries):
    print(f"Query: {text}")
    print(f"Top 5 images:")
    ipyplot.plot_images([Image.open(image_features.indices[i]) for i in indices[k]], [f"{v*100:.1f}%" for v in values[k]], img_width=250)
```
