import numpy as np
import pytest
import torch
import clip

from mlmodule.contrib.clip import CLIPResNet50TextEncoder, CLIPResNet101TextEncoder, CLIPResNet50x4TextEncoder, \
    CLIPViTB32TextEncoder
from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.data.base import IndexedDataset


CLIP_TEXT_MODULE_MAP = {
    x.clip_model_name: x
    for x in (CLIPResNet50TextEncoder, CLIPResNet101TextEncoder, CLIPResNet50x4TextEncoder, CLIPViTB32TextEncoder)
}


@pytest.fixture(params=clip.available_models())
def clip_model_name(request):
    return request.param


def test_state_dict(torch_device, clip_model_name):
    model: torch.nn.Module
    model, _ = clip.load(clip_model_name, device=torch_device)
    ml_clip: BaseTorchMLModule = CLIP_TEXT_MODULE_MAP[clip_model_name](device=torch_device)
    ml_clip.load_state_dict(ml_clip.get_default_pretrained_state_dict_from_provider())

    dict1 = model.state_dict()
    dict2 = ml_clip.state_dict()
    dict1 = {key: dict1[key] for key in dict2}

    assert {k: v.sum() for k, v in dict1.items()} == \
           {k: v.sum() for k, v in dict2.items()}


def test_text_encoding(torch_device, clip_model_name):
    data = ["a dog", "a cat"]

    # Getting the encoded data from Clip
    model, preprocess = clip.load(clip_model_name, device=torch_device, jit=False)
    text = clip.tokenize(data)
    with torch.no_grad():
        clip_output = model.encode_text(text).cpu().numpy()

    # Getting encoded data from MLModule CLIP
    ml_clip: BaseTorchMLModule = CLIP_TEXT_MODULE_MAP[clip_model_name](device=torch_device)
    # TODO: Change to normal load
    ml_clip.load_state_dict(ml_clip.get_default_pretrained_state_dict_from_provider())
    idx, ml_clip_output = ml_clip.bulk_inference(IndexedDataset(list(range(len(data))), data))

    np.testing.assert_allclose(clip_output, ml_clip_output, rtol=1e-2)
