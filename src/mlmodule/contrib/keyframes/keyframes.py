import dataclasses
from typing import List

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin


@dataclasses.dataclass
class KeyFramesExtractor:
    # Controls the minimum threshold between to keyframes
    min_features_distance: float = 14

    @staticmethod
    def kmeans_fit(n_clusters: int, features: np.ndarray) -> KMeans:
        """Fits a KMeans on the given features"""
        return KMeans(n_clusters=n_clusters, algorithm="full", max_iter=20).fit(
            features
        )

    @staticmethod
    def silhouette_stop(scores, steps=3):
        """Tell whether we reached the peak in the silhouette scores"""
        if len(scores) < steps + 1:
            return False
        for i in range(steps):
            # Tries to detect an upward slope in the past "steps"
            if scores[-(i + 1)] - scores[-(i + 2)] > 0:
                return False
        return True

    def find_number_of_frame_clusters(self, features: np.ndarray) -> int:
        """Find the number of cluster using the silhouette method on KMeans"""
        scores: List[float] = []
        for n_clusters in range(2, len(features)):
            model = self.kmeans_fit(n_clusters, features)
            scores.append(silhouette_score(features, model.labels_))
            if self.silhouette_stop(scores):
                break
        return int(np.argmax(scores)) + 2

    def extract_centroid_features(self, features: np.ndarray) -> np.ndarray:
        """Extract centroid features at the center of frames clusters

        If all frames are similar: returns the closest frame to the average frame
        Otherwise: we return the centroid of a KMeans clustering

        Returns
            np.ndarray: dimensions=(n_centroids, features_length)
        """
        max_pairwise_features_distance = pdist(features).max()
        if max_pairwise_features_distance < self.min_features_distance:
            # We consider there is one cluster and return the mean frame
            return np.mean(features, axis=0)[np.newaxis, :]
        elif len(features) == 2:
            # Not enough frames to run KMeans, we keep all frames since they are different
            return features

        # Otherwise, we look from the number of clusters
        num_clusters = self.find_number_of_frame_clusters(features)

        # We get the optimal kmeans and the centroids
        model = self.kmeans_fit(num_clusters, features)
        return model.cluster_centers_

    def extract_keyframes(self, features: np.ndarray) -> List[int]:
        """From frames features returns the list of indices of the keyframes"""
        if len(features) == 1:
            # Only on feature -> return one frame
            return [0]

        centroids = self.extract_centroid_features(features)

        # Finding the unique closest frame for each centroid
        keyframes_indices: List[int] = list(
            set(pairwise_distances_argmin(centroids, features))
        )

        if len(keyframes_indices) == 1:
            return keyframes_indices

        # Checking that the features pass the minimum distance threshold
        max_pairwise_key_features_distance = pdist(features[keyframes_indices]).max()
        if max_pairwise_key_features_distance < self.min_features_distance:
            # Return the closest frame to the average key features
            return list(
                pairwise_distances_argmin(
                    np.mean(features[keyframes_indices], axis=0)[np.newaxis, :],
                    features,
                )
            )

        return keyframes_indices
