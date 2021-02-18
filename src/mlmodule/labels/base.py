from mlmodule.labels.imagenet import IMAGENET_LABELS
from mlmodule.labels.places import PLACES_LABELS
from mlmodule.labels.places_io import PLACES_IO_LABELS


class LabelSet(object):
    # Must uniquely define this label set
    __label_set_name__ = None

    def __init__(self, label_list: list):
        self.label_list = label_list

    def __getitem__(self, item):
        return self.label_list[item]

    def __len__(self):
        return len(self.label_list)


class ImageNetLabels(LabelSet):
    __label_set_name__ = 'imagenet'

    def __init__(self):
        super().__init__([IMAGENET_LABELS[i] for i in range(len(IMAGENET_LABELS))])


class PlacesLabels(LabelSet):
    __label_set_name__ = 'places'

    def __init__(self):
        super().__init__([PLACES_LABELS[i] for i in range(len(PLACES_LABELS))])


class PlacesIOLabels(LabelSet):
    __label_set_name__ = 'places_io'

    def __init__(self):
        super().__init__([PLACES_IO_LABELS[i] for i in range(len(PLACES_IO_LABELS))])


class LabelsMixin(object):

    def get_labels(self) -> LabelSet:
        raise NotImplementedError()
