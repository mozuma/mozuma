import os
from typing import Tuple

import clip
import numpy as np
import pytest
import torch
from PIL import Image

from mozuma.callbacks.memory import CollectFeaturesInMemory
from mozuma.models.clip.image import CLIPImageModule
from mozuma.models.clip.pretrained import (
    torch_clip_image_encoder,
    torch_clip_text_encoder,
)
from mozuma.models.clip.text import CLIPTextModule
from mozuma.torch.datasets import ImageDataset, ListDataset, LocalBinaryFilesDataset
from mozuma.torch.options import TorchRunnerOptions
from mozuma.torch.runners import TorchInferenceRunner


@pytest.fixture(params=["RN50", "ViT-B/32"])
def clip_test_models(
    request,
    torch_device: torch.device,
) -> Tuple[CLIPImageModule, CLIPTextModule]:
    image = torch_clip_image_encoder(request.param, device=torch_device)
    text = torch_clip_text_encoder(request.param, device=torch_device)

    return image, text


def test_text_encoding(clip_test_models: Tuple[CLIPImageModule, CLIPTextModule]):
    data = ["a dog", "a cat"]
    _, ml_clip = clip_test_models

    # Getting the encoded data from Clip
    model, _ = clip.load(ml_clip.clip_model_name, device=ml_clip.device, jit=False)
    text = clip.tokenize(data).to(ml_clip.device)
    with torch.no_grad():
        clip_output = model.encode_text(text).cpu().numpy()

    # Getting encoded data from MoZuMa CLIP
    dataset = ListDataset(data)
    features = CollectFeaturesInMemory()
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=ml_clip,
        callbacks=[features],
        options=TorchRunnerOptions(device=ml_clip.device),
    )
    runner.run()

    np.testing.assert_allclose(clip_output, features.features, rtol=1e-2)


def test_image_encoding(clip_test_models: Tuple[CLIPImageModule, CLIPTextModule]):
    ml_model, _ = clip_test_models
    file_names = [os.path.join("tests", "fixtures", "cats_dogs", "cat_0.jpg")]

    # Getting the encoded data from CLIP
    model, preprocess = clip.load(
        ml_model.clip_model_name, device=ml_model.device, jit=False
    )
    image = preprocess(Image.open(file_names[0])).unsqueeze(0).to(ml_model.device)
    with torch.no_grad():
        clip_output = model.encode_image(image).cpu().numpy()

    # Getting the encoded data from MoZuMa CLIP
    dataset = ImageDataset(LocalBinaryFilesDataset(file_names))
    features = CollectFeaturesInMemory()
    runner = TorchInferenceRunner(
        dataset=dataset,
        model=ml_model,
        callbacks=[features],
        options=TorchRunnerOptions(device=ml_model.device),
    )
    runner.run()

    np.testing.assert_allclose(clip_output, features.features, rtol=1e-2)
