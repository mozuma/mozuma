import numpy as np
import pytest
from sklearn.datasets import make_blobs

from mlmodule.contrib.keyframes.keyframes import KeyFramesExtractor


def test_find_number_of_clusters():
    features, _ = make_blobs(
        n_samples=100, n_features=2, centers=[(5, 5), (5, -5), (-5, 5)], random_state=0
    )

    keyframe_extractor = KeyFramesExtractor(min_features_distance=1)
    assert keyframe_extractor.find_number_of_frame_clusters(features) == 3


@pytest.mark.parametrize(
    "centers,min_distance",
    [
        (
            [
                (5, 5),
                (5, -5),
                (-5, 5),
            ],
            1,
        ),
        (
            [
                (5, 5),
            ],
            3,
        ),
    ],
    ids=["multi-cluster", "single-cluster"],
)
def test_extract_centroid_features(centers: list, min_distance: int):
    features, _ = make_blobs(
        n_samples=100, n_features=2, centers=centers, random_state=0, cluster_std=0.1
    )

    keyframe_extractor = KeyFramesExtractor(min_features_distance=min_distance)
    centroids = keyframe_extractor.extract_centroid_features(features)
    assert len(centroids) == len(centers)
    # Test that for each centroid there is a matching theorical center
    for point in np.array(centers):
        assert any(np.sum((point - c) ** 2) < 0.1 for c in centroids)


@pytest.mark.parametrize(
    "centers,min_distance,n_samples",
    [
        (
            [
                (5, 5),
                (5, -5),
                (-5, 5),
            ],
            1,
            100,
        ),
        (
            [
                (5, 5),
            ],
            3,
            100,
        ),
        (
            [
                (5, 5),
            ],
            3,
            1,
        ),
        (
            [
                (5, 5),
                (5, -5),
            ],
            1,
            2,
        ),
        (
            [
                (5, 5),
                (5, -5),
            ],
            1,
            3,
        ),
    ],
    ids=[
        "multi-cluster",
        "single-cluster",
        "one-sample",
        "two-samples",
        "three-samples",
    ],
)
def test_extract_keyframes(centers: list, min_distance: int, n_samples: int):
    features, labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        random_state=0,
        cluster_std=0.1,
    )

    keyframe_extractor = KeyFramesExtractor(min_features_distance=min_distance)
    keyframe_indices = keyframe_extractor.extract_keyframes(features)
    assert len(keyframe_indices) == len(centers)
    # Check that each keyframe belongs to a different cluster
    assert len(set(labels[keyframe_indices])) == len(centers)
