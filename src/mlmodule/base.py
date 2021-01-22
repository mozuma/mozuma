
class BaseMLModule(object):

    @classmethod
    def load(cls, fp=None, **load_options):
        """Loads the model from a file like object

        :param fp: File object to read the model from. If not provided loads a default pretrained model.
        :param load_options: Passed to load_pretrained and load_from_file method.
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
