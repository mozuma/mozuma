from io import BytesIO
from typing import BinaryIO
import pytest
import torch

from mlmodule.contrib.keyframes.factories import KeyFramesInferenceFactory
from mlmodule.contrib.keyframes.datasets import (
    compute_every_param_from_target_fps,
    FPSVideoFrameExtractor
)
from mlmodule.v2.torch.options import TorchRunnerOptions


@pytest.mark.parametrize(
    ('video_fps', 'max_target_fps', 'result'),
    [
        (24., 1, 24),
        (24., 12, 2),
        (24., 48, 1),
        (11., 2, 6),
    ]
)
def test_compute_every_param_from_target_fps(
        video_fps: float,
        max_target_fps: int,
        result: int
):
    assert compute_every_param_from_target_fps(video_fps, max_target_fps) == result


def test_fps_video_extractor(video_file: BinaryIO):
    extractor = FPSVideoFrameExtractor([0], [video_file], 1)
    _, frames = extractor[0]
    assert len(frames) == 2
    assert len(frames[0]) == 83
    assert len(frames[1]) == 83


def test_keyframes_extractor(torch_device: torch.device, video_file: BinaryIO):
    video_file.seek(0)
    dataset = FPSVideoFrameExtractor([0], [video_file], 1)

    inference_runner = KeyFramesInferenceFactory(
        options=TorchRunnerOptions(
            device=torch_device
        )
    ).get_runner()

    indices, video_keyframes = inference_runner.bulk_inference(dataset)
    assert len(indices) == 1
    assert len(video_keyframes) == 1
    assert len(video_keyframes[0]) > 0 and len(video_keyframes[0]) < 20


def test_keyframes_extractor_bad_file(torch_device: torch.device):
    dataset = FPSVideoFrameExtractor([0], [BytesIO(b"bbbbbb")], 1)

    inference_runner = KeyFramesInferenceFactory(
        options=TorchRunnerOptions(
            device=torch_device
        )
    ).get_runner()

    indices, video_keyframes = inference_runner.bulk_inference(dataset)

    assert len(indices) == 1
    assert len(video_keyframes) == 1
    assert len(video_keyframes[0]) == 0