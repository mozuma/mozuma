import abc
from io import BytesIO
from typing import Callable, List

import torch


class TorchMlModule(torch.nn.Module):
    """
    Module for Torch models.
    """

    def set_state(self, state: bytes, **options) -> None:
        map_location = options.get("device")
        state_dict = torch.load(BytesIO(state), map_location=map_location)
        self.load_state_dict(state_dict)

    def get_state(self, **options) -> bytes:
        f = BytesIO()
        torch.save(self.state_dict(), f)
        return f.read()

    @abc.abstractmethod
    def get_dataset_transforms(self) -> List[Callable]:
        """Returns a callable that will by used to tranform input data into a Tensor passed to the forward function"""

    def batch_to_device(batch_data):
        if isinstance(batch, tuple):
            return tuple(send_batch_to_device(b, device) for b in batch)
        elif isinstance(batch, list):
            return [send_batch_to_device(b, device) for b in batch]
        elif hasattr(batch, "to"):
            return batch.to(device)
        else:
            return batch
