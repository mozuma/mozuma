from io import BytesIO

import numpy as np


class LoadDumpMixin(object):

    @classmethod
    def load(cls, fp=None):
        """Loads the model from a file like object

        :param fp: File object to read the model from. If not provided loads a default pretrained model.
        :return:
        """
        raise NotImplementedError()

    def dump(self, fp):
        """Loads model from a file like object

        :param fp:
        :return:
        """
        raise NotImplementedError()


class BaseMLModule(object):

    def bulk_inference(self, data):
        """Performs inference for all the given data points

        :param data:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _numpy_to_binary(cls, np_array: np.ndarray) -> bytes:
        """Serialize a numpy array as bytes

        :param np_array:
        :return:
        """
        b = BytesIO()
        np.save(b, np_array, allow_pickle=False)
        return b.getvalue()

    @classmethod
    def _numpy_from_binary(cls, binary: bytes) -> np.ndarray:
        """Reads a numpy array from bytes

        :param binary:
        :return:
        """
        return np.load(BytesIO(binary))

    @classmethod
    def to_binary(cls, result_item: np.ndarray) -> bytes:
        """Used to store the result of bulk inference in binary format"""
        return cls._numpy_to_binary(result_item)

    @classmethod
    def from_binary(cls, binary_data: bytes) -> np.ndarray:
        # Reading binary array elements
        return cls._numpy_from_binary(binary_data)
