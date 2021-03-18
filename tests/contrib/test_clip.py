import os

import numpy as np
import pytest
import torch
import clip
from PIL import Image

from mlmodule.contrib.clip import CLIPResNet50TextEncoder, CLIPResNet101TextEncoder, CLIPResNet50x4TextEncoder, \
    CLIPViTB32TextEncoder, CLIPViTB32ImageEncoder, CLIPResNet50x4ImageEncoder, CLIPResNet101ImageEncoder, \
    CLIPResNet50ImageEncoder
from mlmodule.contrib.clip.base import BaseCLIPModule
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset
from mlmodule.torch.data.images import ImageDataset

CLIP_MODULE_MAP = {
    "text": {
        x.clip_model_name: x
        for x in (CLIPResNet50TextEncoder, CLIPResNet101TextEncoder, CLIPResNet50x4TextEncoder,
                  CLIPViTB32TextEncoder)
    },
    "image": {
        x.clip_model_name: x
        for x in (CLIPResNet50ImageEncoder, CLIPResNet101ImageEncoder, CLIPResNet50x4ImageEncoder,
                  CLIPViTB32ImageEncoder)
    }
}


@pytest.fixture(params=clip.available_models())
def clip_model_name(request):
    return request.param


@pytest.mark.parametrize('encoder_type', CLIP_MODULE_MAP.keys())
def test_state_dict(torch_device, clip_model_name, encoder_type):
    model: torch.nn.Module
    model, _ = clip.load(clip_model_name, device=torch_device, jit=False)
    ml_clip: BaseTorchMLModule = CLIP_MODULE_MAP[encoder_type][clip_model_name](device=torch_device)
    ml_clip.load_state_dict(ml_clip.get_default_pretrained_state_dict_from_provider())
    ml_clip.to(torch_device)

    dict1 = model.state_dict()
    dict2 = ml_clip.state_dict()
    dict1 = {key: dict1[key] for key in dict2}

    assert {k: v.sum() for k, v in dict1.items()} == \
           {k: v.sum() for k, v in dict2.items()}


def test_text_encoding(torch_device, clip_model_name):
    data = ["a dog", "a cat"]

    # Getting the encoded data from Clip
    model, preprocess = clip.load(clip_model_name, device=torch_device, jit=False)
    text = clip.tokenize(data).to(torch_device)
    with torch.no_grad():
        clip_output = model.encode_text(text).cpu().numpy()

    # Getting encoded data from MLModule CLIP
    ml_clip: BaseCLIPModule = CLIP_MODULE_MAP["text"][clip_model_name](device=torch_device)
    # TODO: Change to normal load
    ml_clip.load_state_dict(ml_clip.get_default_pretrained_state_dict_from_provider())
    idx, ml_clip_output = ml_clip.bulk_inference(IndexedDataset(list(range(len(data))), data))

    np.testing.assert_allclose(clip_output, ml_clip_output, rtol=1e-2)


def test_image_encoding(torch_device, clip_model_name):
    file_names = [os.path.join("tests", "fixtures", "cats_dogs", "cat_0.jpg")]
    dataset = ImageDataset(file_names)

    # Getting the encoded data from CLIP
    model, preprocess = clip.load(clip_model_name, device=torch_device, jit=False)
    image = preprocess(Image.open(file_names[0])).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        clip_output = model.encode_image(image).cpu().numpy()

    # Getting the encoded data from MLModule CLIP
    ml_clip: BaseCLIPModule = CLIP_MODULE_MAP["image"][clip_model_name](device=torch_device)
    # TODO: Change to normal load
    ml_clip.load_state_dict(ml_clip.get_default_pretrained_state_dict_from_provider())
    idx, ml_clip_output = ml_clip.bulk_inference(dataset)

    np.testing.assert_allclose(clip_output, ml_clip_output, rtol=1e-2)
