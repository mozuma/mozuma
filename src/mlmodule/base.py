from io import BytesIO

import numpy as np


class BaseMLModule(object):
    # Used for the to_binary / from_binary.
    # If None, the objects returned by `bulk_inference` will be serialized with numpy.save without pickle
    # Otherwise, it should contain a namedtuple definition, the bulk_inference returns a list of these namedtuple
    # In order for `to_binary` to work, the elements of the namedtuple
    # should be serializable with numpy.save without pickle
    __result_struct__ = None

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

    def bulk_inference(self, data):
        """Performs inference for all the given data points

        :param data:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _numpy_to_binary(cls, np_array: np.array):
        """Serialize a numpy array as bytes

        :param np_array:
        :return:
        """
        b = BytesIO()
        np.save(b, np_array, allow_pickle=False)
        return b.getvalue()

    @classmethod
    def _numpy_from_binary(cls, binary: bytes):
        """Reads a numpy array from bytes

        :param binary:
        :return:
        """
        return np.load(BytesIO(binary))

    @classmethod
    def to_binary(cls, result_item):
        """Used to store the result of bulk inference in binary format

        :param result_item:
        :param result_struct: Optional structure of the result item
        :return:
        """
        if cls.__result_struct__ is not None:
            # Saving each item of the tuple as binary first (to avoid using pickle)
            result_item = np.array([cls._numpy_to_binary(r) for r in result_item])
        # Saving array as binary
        return cls._numpy_to_binary(result_item)

    @classmethod
    def from_binary(cls, binary_data):
        # Reading binary array elements
        result_item = cls._numpy_from_binary(binary_data)
        if cls.__result_struct__ is not None:
            # In the case of namedtuple, the array elements are array in binary format that we need to read
            return cls.__result_struct__(*[cls._numpy_from_binary(r) for r in result_item])
        else:
            return result_item
