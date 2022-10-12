---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.7.12 64-bit
    language: python
    name: python3
---

# Overview of search capabilities

<!-- #region -->

<a target="_blank" href="https://github.com/mozuma/mozuma/blob/master/docs/examples/1-overview.ipynb">
  <img src="https://img.shields.io/static/v1?label=&message=See%20the%20source%20code&color=blue&logo=github&labelColor=black" alt="See the source code"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/mozuma/mozuma/blob/master/docs/examples/1-overview.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


This notebook contains sample code that helps getting started with building queries with MoZuMa.
It shows how to extract embeddings of images with MoZuMa as well as how to filter or rank images using these embeddings.

<!-- #endregion -->

```python
# Install additional requirements
!pip install ipyplot
```

## Downloading images

We create a small collection of images that can be run on CPU.


```python
from PIL import Image, ImageDraw
import requests

IMAGE_URLS = [
    "https://images.pexels.com/photos/13021281/pexels-photo-13021281.jpeg?cs=srgb&dl=pexels-moaz-tobok-13021281.jpg&fm=jpg&w=1280&h=1920",
    "https://images.pexels.com/photos/13168607/pexels-photo-13168607.jpeg?cs=srgb&dl=pexels-valeria-boltneva-13168607.jpg&fm=jpg&w=1280&h=1919",
    "https://images.pexels.com/photos/4327709/pexels-photo-4327709.jpeg?cs=srgb&dl=pexels-jess-loiterton-4327709.jpg&fm=jpg&w=1280&h=1707",
    "https://images.pexels.com/photos/4327792/pexels-photo-4327792.jpeg?cs=srgb&dl=pexels-jess-loiterton-4327792.jpg&fm=jpg&w=1280&h=1600",
    "https://images.pexels.com/photos/3773651/pexels-photo-3773651.jpeg?cs=srgb&dl=pexels-arnie-watkins-3773651.jpg&fm=jpg&w=1280&h=1600",
    "https://images.pexels.com/photos/2962392/pexels-photo-2962392.jpeg?cs=srgb&dl=pexels-symeon-ekizoglou-2962392.jpg&fm=jpg&w=1280&h=1852",
    "https://images.pexels.com/photos/2407089/pexels-photo-2407089.jpeg?cs=srgb&dl=pexels-flynn-grey-2407089.jpg&fm=jpg&w=1280&h=853",
    # Kayak sea cave
    "https://images.pexels.com/photos/2847862/pexels-photo-2847862.jpeg?cs=srgb&dl=pexels-ngoc-vuong-2847862.jpg&fm=jpg&w=1280&h=850",
    # Kayak sea
    "https://images.pexels.com/photos/1430672/pexels-photo-1430672.jpeg?cs=srgb&dl=pexels-asad-photo-maldives-1430672.jpg&fm=jpg&w=1280&h=863",
    # Kayak mountain
    "https://images.pexels.com/photos/1497582/pexels-photo-1497582.jpeg?cs=srgb&dl=pexels-spencer-gurley-films-1497582.jpg&fm=jpg&w=1280&h=854",
    "https://images.pexels.com/photos/13759/pexels-photo-13759.jpeg?cs=srgb&dl=pexels-jamie-hutt-13759.jpg&fm=jpg&w=1280&h=960",
    "https://images.pexels.com/photos/1252396/pexels-photo-1252396.jpeg?cs=srgb&dl=pexels-headshatter-1252396.jpg&fm=jpg&w=1280&h=1920",
    # Kayak helmet
    "https://images.pexels.com/photos/2283103/pexels-photo-2283103.jpeg?cs=srgb&dl=pexels-brett-sayles-2283103.jpg&fm=jpg&w=1280&h=852",
    # Moto
    "https://images.pexels.com/photos/39693/motorcycle-racer-racing-race-speed-39693.jpeg?cs=srgb&dl=pexels-pixabay-39693.jpg&fm=jpg&w=1280&h=850",
    # Dog
    "https://images.pexels.com/photos/58997/pexels-photo-58997.jpeg?cs=srgb&dl=pexels-muhannad-alatawi-58997.jpg&fm=jpg&w=1280&h=853",
    "https://images.pexels.com/photos/1254140/pexels-photo-1254140.jpeg?cs=srgb&dl=pexels-johann-1254140.jpg&fm=jpg&w=1280&h=853",
    "https://images.pexels.com/photos/551628/pexels-photo-551628.jpeg?cs=srgb&dl=pexels-kat-smith-551628.jpg&fm=jpg&w=1280&h=854",
    # Sea cave
    "https://images.pexels.com/photos/163872/italy-cala-gonone-air-sky-163872.jpeg?cs=srgb&dl=pexels-pixabay-163872.jpg&fm=jpg&w=1280&h=960",
]

KAYAK_IMAGE = "https://images.pexels.com/photos/1497582/pexels-photo-1497582.jpeg?cs=srgb&dl=pexels-spencer-gurley-films-1497582.jpg&fm=jpg&w=1280&h=854"
SEA_CAVE_IMAGE = "https://images.pexels.com/photos/163872/italy-cala-gonone-air-sky-163872.jpeg?cs=srgb&dl=pexels-pixabay-163872.jpg&fm=jpg&w=1280&h=960"

images_objects = [Image.open(requests.get(url, stream=True).raw) for url in IMAGE_URLS]
```


## Helper functions

Definition of a few functions to display images or highlight objects in an image

```python
from typing import Iterable

import ipyplot
import numpy as np
import numpy.typing as npt

from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity


def display_images(indices: "Iterable[int] | None" = None, **kwargs):
    indices = indices if indices is not None else range(len(IMAGE_URLS))
    ipyplot.plot_images([IMAGE_URLS[i] for i in indices], **kwargs)


def draw_bounding_box(
    image: Image.Image, bounding_box: npt.NDArray[np.float_]
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bounding_box.tolist(), outline="red", width=10)
    return image


def display_crops(
    image_indices: Iterable[int], bounding_boxes: npt.NDArray[np.float_], **kwargs
):
    cropped_images = [
        draw_bounding_box(images_objects[image_index].copy(), bb)
        for image_index, bb in zip(image_indices, bounding_boxes)
    ]
    ipyplot.plot_images(cropped_images, **kwargs)


def cosine_similarity(
    query: npt.NDArray[np.float_], features: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    if len(query.shape) == 1:
        query = query[np.newaxis]
    return _cosine_similarity(query, features)[0]


def arg_rank_by_cosine_similarity(
    query: npt.NDArray[np.float_],
    features: npt.NDArray[np.float_],
    take: "int | None" = None,
) -> npt.NDArray[np.int_]:
    result = np.argsort(cosine_similarity(query, features))[::-1]
    if take is not None:
        return result[:take]
    return result
```

## The image collection

```python
display_images()
```

## Basic interface to run models

Generic code to run a model inference on PyTorch.


```python
import torch
from mozuma.torch.callbacks import TorchRunnerCallbackType
from mozuma.torch.datasets import ListDatasetIndexed, TorchDataset
from mozuma.torch.modules import TorchModel
from mozuma.torch.runners import TorchInferenceRunner
from mozuma.torch.options import TorchRunnerOptions


def run_torch_model_inference(
    model: TorchModel,
    callbacks: "list[TorchRunnerCallbackType]",
    dataset: "TorchDataset | None" = None,
) -> None:
    """Runs inference for a PyTorch model"""
    # Setting the dataset to images if not defined
    dataset = dataset or ListDatasetIndexed(indices=IMAGE_URLS, objects=images_objects)

    runner = TorchInferenceRunner(
        model=model,
        dataset=dataset,
        callbacks=callbacks,
        options=TorchRunnerOptions(
            device=torch.device("cpu"), data_loader_options={"batch_size": 20}
        ),
    )
    runner.run()
```

## Generic function to compute features


```python
from mozuma.callbacks import CollectFeaturesInMemory


def collect_features(
    model: TorchModel, dataset: "TorchDataset | None" = None
) -> npt.NDArray[np.float_]:
    features = CollectFeaturesInMemory()
    run_torch_model_inference(model=model, callbacks=[features], dataset=dataset)
    if dataset is None:
        assert features.indices == IMAGE_URLS, features.indices
    return features.features
```

## Find an image from a text query

We are looking at images matching the text query: `A dog at the beach`


```python
from mozuma.models.clip.pretrained import (
    torch_clip_image_encoder,
    torch_clip_text_encoder,
)
from mozuma.torch.datasets import ListDataset


clip_image_features = collect_features(model=torch_clip_image_encoder("ViT-B/32"))
clip_text_features = collect_features(
    model=torch_clip_text_encoder("ViT-B/32"),
    dataset=ListDataset(["a dog at the beach"]),
)
display_images(
    arg_rank_by_cosine_similarity(clip_text_features, clip_image_features, take=1),
    img_width=500,
)
```


# Generic function to compute bounding boxes


```python
from mozuma.callbacks import CollectBoundingBoxesInMemory


def collect_bbox(
    model: TorchModel,
) -> "tuple[npt.NDArray[np.str_], npt.NDArray[np.float_], npt.NDArray[np.float_]]":
    bbox = CollectBoundingBoxesInMemory()
    run_torch_model_inference(model=model, callbacks=[bbox])
    assert bbox.indices == IMAGE_URLS, bbox.indices
    # Flattening the bounding boxes
    indices: "list[str]" = []
    features: "list[npt.NDArray[np.float_]]" = []
    boxes: "list[npt.NDArray[np.float_]]" = []
    for index, box_list in zip(bbox.indices, bbox.bounding_boxes):
        indices += [index] * len(box_list.bounding_boxes)
        boxes.append(box_list.bounding_boxes)
        if box_list.features is None:
            raise ValueError("This model does not returned features")
        features.append(box_list.features)
    return np.array(indices, dtype=str), np.vstack(boxes), np.vstack(features)
```

## Find images with similar objects

We are looking for image with a paddle.

First, we need to run object detection on all images. This can take 10-15 minutes


```python
from mozuma.models.vinvl.pretrained import torch_vinvl_detector

bbox_indices, bbox_boxes, bbox_features = collect_bbox(model=torch_vinvl_detector())
```


Then, we retrieve the features of a detected paddle object

```python
from scipy.spatial.distance import cdist

# Find an image of a paddle
paddle_coordinates = np.array([899.95416, 581.6102, 1105.5442, 640.5274])
paddle_box_index = np.argmin(
    cdist(paddle_coordinates[np.newaxis], bbox_boxes[bbox_indices == KAYAK_IMAGE])
)
paddle_bounding_box = bbox_boxes[bbox_indices == KAYAK_IMAGE][paddle_box_index]
paddle_features = bbox_features[bbox_indices == KAYAK_IMAGE][paddle_box_index]

display_crops(
    [IMAGE_URLS.index(KAYAK_IMAGE)], paddle_bounding_box[np.newaxis], img_width=500
)
```


Finally, we rank object by similarity to the paddle object passed as query.

```python
# Finding similar objects
top_matching_objects = arg_rank_by_cosine_similarity(
    paddle_features, bbox_features, take=12
)[1:]
top_matching_objects_image_urls = bbox_indices[top_matching_objects]

display_crops(
    [IMAGE_URLS.index(img_url) for img_url in top_matching_objects_image_urls],
    bbox_boxes[top_matching_objects, :],
    img_width=500,
)
```


# Combining objects and scene similarity queries

First we get the scene recognition features for all images

```python
from mozuma.models.densenet.pretrained import torch_densenet_places365

densenet_places_features = collect_features(model=torch_densenet_places365())
```

Display the query image for scene similarity

```python
display_images([IMAGE_URLS.index(SEA_CAVE_IMAGE)], img_width=500)
```

Finally, we construct the query logic:

1. Filter all images with a paddle (threshold of 0.5 on cosine similarity).
1. Rank image by similarity to the query scene
1. Take the first image that matches the paddle object in the list of ranked images by similarity to the scene


```python
# Find images with an object that looks like a paddle with threshold 0.5 on cosine similarity
paddle_objects = cosine_similarity(paddle_features, bbox_features) > 0.5
image_urls_with_paddles = set(bbox_indices[paddle_objects])

# Ranking image with paddles with the places365 similarity
display_images(
    [
        img_idx
        for img_idx in arg_rank_by_cosine_similarity(
            densenet_places_features[IMAGE_URLS.index(SEA_CAVE_IMAGE)],
            densenet_places_features,
        )
        if IMAGE_URLS[img_idx] in image_urls_with_paddles
    ][:1],
    img_width=500,
)
```
