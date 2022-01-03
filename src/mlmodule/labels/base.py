from typing import List

from mlmodule.labels.imagenet import IMAGENET_LABELS
from mlmodule.labels.places import PLACES_LABELS
from mlmodule.labels.places_io import PLACES_IN_OUT_DOOR, PLACES_IO_LABELS
from mlmodule.labels.vinvl import VINVL_LABELS
from mlmodule.labels.vinvl_attributes import VINVL_ATTRIBUTE_LABELS


class LabelSet(object):
    # Must uniquely define this label set
    __label_set_name__: str = None
    label_list: List[str] = None

    def __getitem__(self, item):
        return self.label_list[item]

    def __len__(self):
        return len(self.label_list)


class ImageNetLabels(LabelSet):
    __label_set_name__ = "imagenet"
    label_list = IMAGENET_LABELS


class PlacesLabels(LabelSet):
    __label_set_name__ = "places"
    label_list = PLACES_LABELS


class PlacesIOLabels(LabelSet):
    __label_set_name__ = "places_io"
    label_list = PLACES_IO_LABELS


class IndoorOutdoorLabels(LabelSet):
    __label_set_name__ = "in_out_door"
    label_list = PLACES_IN_OUT_DOOR


class VinVLLabels(LabelSet):
    __label_set_name__ = "vinvl"
    label_list = VINVL_LABELS


class VinVLAttributeLabels(LabelSet):
    __label_set_name__ = "vinvl_attributes"
    label_list = VINVL_ATTRIBUTE_LABELS


class LabelsMixin(object):
    def get_labels(self) -> LabelSet:
        raise NotImplementedError()
