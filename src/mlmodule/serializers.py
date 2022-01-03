import dataclasses
from typing import List, Union

import numpy as np

from mlmodule.box import BBoxCollection, BBoxOutput
from mlmodule.frames import FrameOutput, FrameOutputCollection

_BBoxModelOutput = List[BBoxCollection]
_FramesModelOutput = List[FrameOutputCollection]
_FeaturesModelOutput = np.ndarray


@dataclasses.dataclass
class Serializer:
    data: Union[_BBoxModelOutput, _FeaturesModelOutput, _FramesModelOutput]

    @classmethod
    def safe_json_bbox(cls, bbox: BBoxOutput) -> dict:
        """Turns the BBoxOutput dataclass into a dictionnary that is JSON serializable"""
        bbox_dict = dataclasses.asdict(bbox)
        if bbox_dict["features"] is not None:
            bbox_dict["features"] = bbox_dict["features"].tolist()
        return bbox_dict

    @classmethod
    def safe_json_frames(cls, frame: FrameOutput) -> dict:
        """Turns the FrameOutput dataclass into a dictionnary"""
        frame_dict = dataclasses.asdict(frame)
        if frame_dict["features"] is not None:
            frame_dict["features"] = frame_dict["features"].tolist()
        return frame_dict

    def safe_json_types(self) -> list:
        """Returns the data attribute as a dictionnary"""
        if (
            type(self.data) == list
            and len(self.data) > 0
            and type(self.data[0]) == list
            and len(self.data[0]) > 0
        ):
            if isinstance(self.data[0][0], BBoxOutput):
                # This is a list of BoudingBox collection
                return [
                    [self.safe_json_bbox(bbox) for bbox in col] for col in self.data
                ]
            elif isinstance(self.data[0][0], FrameOutput):
                return [
                    [self.safe_json_frames(frame) for frame in col] for col in self.data
                ]
        elif hasattr(self.data, "tolist"):
            # This is a numpy array
            return self.data.tolist()
        else:
            return self.data
