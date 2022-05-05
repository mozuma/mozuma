import numpy as np

from mlmodule.callbacks.memory import CollectFeaturesInMemory


def test_collect_features():
    callback = CollectFeaturesInMemory()

    # Calling the save features twice
    callback.save_features(None, [1, 2], np.array([[1, 1, 1], [2, 2, 2]]))
    callback.save_features(
        None, [5, 6], np.array([[5, 5, 5], [np.nan, np.nan, np.nan]])
    )

    # Checking the state
    assert callback.indices == [1, 2, 5, 6]
    np.testing.assert_equal(
        callback.features,
        np.array([[1, 1, 1], [2, 2, 2], [5, 5, 5], [np.nan, np.nan, np.nan]]),
    )
