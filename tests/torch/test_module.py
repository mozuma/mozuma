from typing import Any, Dict, Optional, Sequence

import pytest
import torch

from mlmodule.torch.modules import TorchMlModule

_CPU = torch.device("cpu")


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        ((float, _CPU), {"other": 13}, _CPU),
        ((0, int), {"other": 13}, torch.device(0)),
        ((float, int), {"other": 13, "device": _CPU}, _CPU),
        ((float, int), {"other": 13}, None),
    ],
    ids=["args-device", "args-int", "kwargs", "not-passed"],
)
def test_extract_device_from_args(
    args: Sequence, kwargs: Dict[str, Any], expected: Optional[torch.device]
):
    assert TorchMlModule._extract_device_from_args(*args, **kwargs) == expected
