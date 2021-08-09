import os

from mlmodule.contrib.rpn.base import RegionFeatures
from mlmodule.contrib.rpn.rpn import RPN
from mlmodule.contrib.rpn.selector import CosineSimilarityRegionSelector
from mlmodule.contrib.rpn.encoder import RegionEncoder, DenseNet161ImageNetEncoder


def get_absolute_config_path(relative_config_path: str) -> str:
    """Translates a relative config path to an absolute one"""
    return os.path.join(os.path.dirname(__file__), relative_config_path)
