from collections import namedtuple

import numpy as np
from mlmodule.base import BaseMLModule


class NumpyFeaturesModule(BaseMLModule):
    __result_struct__ = None


class NamedTuplesFeaturesModule(BaseMLModule):
    __result_struct__ = namedtuple('NT', ['boxes', 'data'])


def test_to_binary_numpy():
    # Encoding / decoding and assert equals
    src = np.random.rand(512)
    binary = NumpyFeaturesModule.to_binary(src)
    dec = NumpyFeaturesModule.from_binary(binary)
    np.testing.assert_equal(src, dec)


def test_to_binary_named_tuple():
    src_box = np.random.rand(10)
    src_data = np.random.rand(1000)
    feature = NamedTuplesFeaturesModule.__result_struct__(src_box, src_data)
    binary = NamedTuplesFeaturesModule.to_binary(feature)
    dec = NamedTuplesFeaturesModule.from_binary(binary)

    # Comparing keys
    np.testing.assert_equal(src_box, dec.boxes)
    np.testing.assert_equal(src_data, dec.data)
