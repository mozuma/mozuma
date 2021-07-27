import dataclasses
from typing import List, Union

import numpy as np

from mlmodule.box import BBoxCollection, BBoxOutput


_BBoxModelOutput = List[BBoxCollection]
_FeaturesModelOutput = np.ndarray


@dataclasses.dataclass
class Serializer:
    data: Union[_BBoxModelOutput, _FeaturesModelOutput]

    @classmethod
    def safe_json_bbox(cls, bbox: BBoxOutput) -> dict:
        """Turns the BBoxOutput dataclass into a dictionnary that is JSON serializable"""
        bbox_dict = dataclasses.asdict(bbox)
        if bbox_dict["features"] is not None:
            bbox_dict["features"] = bbox_dict["features"].tolist()
        return bbox_dict

    def safe_json_types(self) -> list:
        """Returns the data attribute as a dictionnary"""
        if type(self.data) == list and len(self.data) > 0 and \
            type(self.data[0]) == list and len(self.data[0]) > 0 and type(self.data[0][0]) == BBoxOutput:
            # This is a list of BoudingBox collection
            return [[self.safe_json_bbox(bbox) for bbox in col] for col in self.data]
        else:
            # This is a numpy array
            return self.data.tolist()
