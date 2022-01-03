from io import BytesIO
from typing import Callable, Union

from mlmodule.torch.base import BaseTorchMLModule
from mlmodule.torch.mixins import DownloadPretrainedStateFromProvider
from mlmodule.types import StateDict


def test_pretrained_download_from_provider(
    provider_pretrained_module: Union[
        BaseTorchMLModule, DownloadPretrainedStateFromProvider
    ],
    assert_state_dict_equals: Callable[[StateDict, StateDict], None],
) -> None:
    model: BaseTorchMLModule = provider_pretrained_module(device="cpu")
    model.load_state_dict(model.get_default_pretrained_state_dict_from_provider())
    buf = BytesIO()
    model.dump(buf)
    buf.seek(0)

    other_model: BaseTorchMLModule = provider_pretrained_module(device="cpu").load(buf)

    assert_state_dict_equals(model.state_dict(), other_model.state_dict())
