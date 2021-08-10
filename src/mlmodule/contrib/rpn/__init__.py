import os

from mlmodule.contrib.rpn.base import RegionFeatures


__all__ = ['RegionFeatures']


def get_absolute_config_path(relative_config_path: str) -> str:
    """Translates a relative config path to an absolute one"""
    return os.path.join(os.path.dirname(__file__), relative_config_path)
