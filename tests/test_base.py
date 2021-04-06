import numpy as np
from mlmodule.base import BaseMLModule


def test_to_binary_numpy():
    # Encoding / decoding and assert equals
    src = np.random.rand(512)
    binary = BaseMLModule.to_binary(src)
    dec = BaseMLModule.from_binary(binary)
    np.testing.assert_equal(src, dec)
