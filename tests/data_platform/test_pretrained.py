from typing import Union

from mlmodule.torch import BaseTorchMLModule
from mlmodule.torch.mixins import TorchPretrainedModuleMixin


def test_deterministic_pretrained_dict(
        data_platform_scanner: Union[BaseTorchMLModule, TorchPretrainedModuleMixin]
):
    # Module should consistently return the same pretrained dictionary
    dict1 = data_platform_scanner().get_default_pretrained_state_dict()
    dict2 = data_platform_scanner().get_default_pretrained_state_dict()

    assert {k: v.sum() for k, v in dict1.items()} == \
           {k: v.sum() for k, v in dict2.items()}
