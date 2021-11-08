import abc
from io import BytesIO
from typing import Callable, List

import torch


class TorchModel(torch.nn.Module):

    def set_state(self, state: bytes, **options) -> None:
        map_location = options.get('device')
        state_dict = torch.load(
            BytesIO(state),
            map_location=map_location
        )
        self.load_state_dict(state_dict)

    def get_state(self, **options) -> bytes:
        f = BytesIO()
        torch.save(self.state_dict(), f)
        return f.read()

    @abc.abstractmethod
    def get_dataset_transforms(self) -> List[Callable]:
        """Returns a callable that will by used to tranform input data into a Tensor passed to the forward function"""
