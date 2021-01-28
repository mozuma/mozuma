import pytest
import torch


@pytest.fixture
def device(request):
    if request.param != 'cpu' and not torch.cuda.is_available():
        pytest.skip(f"Skipping device {request.param}, CUDA not available")
    return torch.device(request.param)


device_parametrize = pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
