from typing import Tuple, List

from mlmodule.base import BaseMLModule
from mlmodule.box import BBoxCollection


class RegionFeatures(BaseMLModule):

    def __init__(self, rpn, region_encoder, region_selector):
        self.rpn = rpn
        self.region_encoder = region_encoder
        self.region_selector = region_selector

    def bulk_inference(self, data) -> Tuple[List, List[BBoxCollection]]:
        """Performs inference for all the given data points

        :param data:
        :return:
        """
        # TODO
        raise NotImplementedError
