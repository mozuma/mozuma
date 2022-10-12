---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: 'Python 3.7.10 64-bit (''mozuma'': conda)'
    name: python3
---

# Video Key-frames Extraction Model



<a target="_blank" href="https://github.com/mozuma/mozuma/blob/master/docs/examples/keyframes.ipynb">
  <img src="https://img.shields.io/static/v1?label=&message=See%20the%20source%20code&color=blue&logo=github&labelColor=black" alt="See the source code"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/mozuma/mozuma/blob/master/docs/examples/keyframes.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Import `mozuma` modules

```python
from mozuma.torch.datasets import (
    LocalBinaryFilesDataset,
)
from mozuma.models.keyframes.pretrained import torch_keyframes_resnet_imagenet
from mozuma.models.keyframes.datasets import (
    BinaryVideoCapture,
    extract_video_frames,
)
from mozuma.callbacks.memory import CollectVideoFramesInMemory
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner

import torch
import os
```

Load a test video

```python
dataset = LocalBinaryFilesDataset(
    [os.path.join("../../tests", "fixtures", "video", "test.mp4")]
)
```


Extract key-frames with `torch_keyframes_resnet_imagenet`

```python
torch_device = torch.device("cpu")
# define model for keyframes extractor
model = torch_keyframes_resnet_imagenet("resnet18", device=torch_device)

features = CollectVideoFramesInMemory()
runner = TorchInferenceRunner(
    model=model,
    dataset=dataset,
    callbacks=[features],
    options=TorchRunnerOptions(
        device=torch_device, data_loader_options={"batch_size": 1}, tqdm_enabled=True
    ),
)
runner.run()
```

Visualise the extracted key-frames


First install `ipyplot`

```python
# Install a pip package in the current Jupyter kernel
import sys

!{sys.executable} -m pip install ipyplot
```

Then extract the selected key-frames from the video

```python
with open(
    os.path.join("../../tests", "fixtures", "video", "test.mp4"), mode="rb"
) as video_file:
    with BinaryVideoCapture(video_file) as capture:
        video_frames = dict(extract_video_frames(capture))
frame_positions = sorted(kf for kf in features.frames[0].frame_indices)
selected_frames = [video_frames[i] for i in frame_positions]
```

Finally, display the frames

```python
import ipyplot

ipyplot.plot_images(selected_frames, frame_positions, img_width=250)
```


```python

```
