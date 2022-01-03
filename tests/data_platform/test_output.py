from typing import Set, Type, Union

import pytest
import torch

from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.data.images import ImageDataset


def test_output_indices(
    image_module: Union[BaseTorchMLModule],
    image_dataset: ImageDataset,
    torch_device: torch.device,
    gpu_only_modules: Set[Type[BaseTorchMLModule]],
) -> None:
    saved_transforms = image_dataset.transforms.copy()
    try:
        if image_module in gpu_only_modules and torch_device == torch.device("cpu"):
            pytest.skip(f"Skipping module {image_module} as it needs to run on GPU")
        ml = image_module(device=torch_device)
        indices, _ = ml.bulk_inference(image_dataset)

        assert set(indices) == set(image_dataset.indices)
    except Exception:
        raise
    finally:
        image_dataset.transforms = saved_transforms
