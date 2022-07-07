import functools
from typing import Any, Iterable, List, Optional, Union

import numpy as np

from mozuma.predictions import (
    BatchBoundingBoxesPrediction,
    BatchModelPrediction,
    BatchVideoFramesPrediction,
)


@functools.singledispatch
def _predictions_data_serializer(data) -> Optional[Union[dict, list]]:
    """Recursively serializes data contained in of a batch model prediction object"""
    return data


@_predictions_data_serializer.register
def _(data: np.ndarray) -> list:
    return data.tolist()


@_predictions_data_serializer.register
def _(data: list) -> list:
    return [_predictions_data_serializer(d) for d in data]


@_predictions_data_serializer.register
def _(data: None) -> None:
    return None


@_predictions_data_serializer.register
def _(data: BatchBoundingBoxesPrediction) -> dict:
    return {
        "bounding_boxes": _predictions_data_serializer(data.bounding_boxes),
        "scores": _predictions_data_serializer(data.scores),
        "features": _predictions_data_serializer(data.features),
    }


@_predictions_data_serializer.register
def _(data: BatchVideoFramesPrediction) -> dict:
    return {
        "features": _predictions_data_serializer(data.features),
        "frame_indices": _predictions_data_serializer(data.frame_indices),
    }


def get_batch_model_prediction_attribute_at_index(
    batch_model_prediction: BatchModelPrediction[np.ndarray],
    attribute: str,
    index: int,
) -> Optional[
    Union[np.ndarray, BatchVideoFramesPrediction, BatchBoundingBoxesPrediction]
]:
    """Returns for the given object attribute the data point at index"""
    attribute_value = getattr(batch_model_prediction, attribute)
    return attribute_value[index] if attribute_value is not None else None


def batch_model_prediction_to_dict(
    indices: Iterable[Any], predictions: BatchModelPrediction[np.ndarray]
) -> List[dict]:
    ret: List[dict] = []
    for index, index_value in enumerate(indices):
        d = {"index": index_value}
        for attribute in ("features", "label_scores", "frames", "bounding_boxes"):
            d[attribute] = _predictions_data_serializer(
                get_batch_model_prediction_attribute_at_index(
                    predictions, attribute, index
                )
            )
        ret.append(d)

    return ret
