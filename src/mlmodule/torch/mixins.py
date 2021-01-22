from torchvision.transforms import Compose


class TorchPretrainedModuleMixin(object):

    def get_default_pretrained_state_dict(self, map_location=None, cache_dir=None, **options):
        """
        Returns the state dict to apply to the current module to get a pretrained model
        :return:
        """
        raise NotImplementedError()


class TorchDatasetTransformsMixin(object):

    def add_transforms(self, transforms):
        """Adding transforms to the list

        :param transforms:
        :return:
        """
        self.transforms += transforms

    def apply_transforms(self, x):
        """Applies the list of transforms to x

        :param x:
        :return:
        """
        return Compose(self.transforms)(x)
