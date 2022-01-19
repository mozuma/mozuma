from io import BytesIO
from typing import BinaryIO

import pytest
import torch

from mlmodule.contrib.keyframes.datasets import (
    FPSVideoFrameExtractorTransform,
    compute_every_param_from_target_fps,
)
from mlmodule.contrib.keyframes.factories import KeyFramesInferenceFactory
from mlmodule.contrib.keyframes.v1 import TorchMLModuleKeyFrames
from mlmodule.v2.torch.datasets import ListDataset
from mlmodule.v2.torch.options import TorchRunnerOptions


@pytest.mark.parametrize(
    ("video_fps", "max_target_fps", "result"),
    [
        (24.0, 1, 24),
        (24.0, 12, 2),
        (24.0, 48, 1),
        (11.0, 2, 6),
    ],
)
def test_compute_every_param_from_target_fps(
    video_fps: float, max_target_fps: int, result: int
):
    assert compute_every_param_from_target_fps(video_fps, max_target_fps) == result


def test_fps_video_extractor(video_file_path: str):
    with open(video_file_path, mode="rb") as video_file:
        frames = FPSVideoFrameExtractorTransform(fps=1)(video_file)
    assert len(frames) == 2
    assert len(frames[0]) == 83
    assert len(frames[1]) == 83


def test_keyframes_extractor(torch_device: torch.device, video_file_path: str):
    with open(video_file_path, mode="rb") as video_file:
        dataset = ListDataset([video_file])

        inference_runner = KeyFramesInferenceFactory(
            options=TorchRunnerOptions(device=torch_device)
        ).get_runner()

        indices, video_keyframes = inference_runner.bulk_inference(dataset)
        assert len(indices) == 1
        assert len(video_keyframes) == 1
        assert len(video_keyframes[0]) > 0 and len(video_keyframes[0]) < 21


def test_keyframes_extractor_v1(torch_device: torch.device, video_file_path: str):
    with open(video_file_path, mode="rb") as video_file:
        dataset = ListDataset([video_file])

        model = TorchMLModuleKeyFrames(device=torch_device).load()

        indices, video_keyframes = model.bulk_inference(dataset)
        assert len(indices) == 1
        assert len(video_keyframes) == 1
        assert len(video_keyframes[0]) > 0 and len(video_keyframes[0]) < 20


def test_keyframes_extractor_bad_file(torch_device: torch.device):
    dataset = ListDataset([BytesIO(b"bbbbbb")])

    inference_runner = KeyFramesInferenceFactory(
        options=TorchRunnerOptions(device=torch_device)
    ).get_runner()

    indices, video_keyframes = inference_runner.bulk_inference(dataset)

    assert len(indices) == 1
    assert len(video_keyframes) == 1
    assert len(video_keyframes[0]) == 0
