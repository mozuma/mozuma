from typing import Callable, Dict, Union
from urllib3.packages.six import BytesIO

import numpy as np
from torch.tensor import Tensor

from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin
from mlmodule.types import StateDict


def test_load(
    data_platform_scanner: Union[BaseTorchMLModule, TorchPretrainedModuleMixin],
    assert_state_dict_equals: Callable[[StateDict, StateDict], None]
) -> None:
    """Test that a scanner can be loaded and dumped"""
    # Extract models weights
    model: BaseTorchMLModule = data_platform_scanner(device='cpu').load()
    buf = BytesIO()
    model.dump(buf)
    buf.seek(0)

    other_model: BaseTorchMLModule = data_platform_scanner(device='cpu').load(buf)

    assert_state_dict_equals(model.state_dict(), other_model.state_dict())
