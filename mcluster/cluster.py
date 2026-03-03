import numpy as np
import mlx.core as mx
from rdkit.ML.Cluster import Butina
from typing import Optional


def fp_to_mlx(fp: list) -> mx.array:
    """
    Convert a list of fingerprints into MLX array
    :param fp: list
        A list of molecular fingerprints calcualted using RDKit.
    :return:
    """
    # convert list to array then to mx.array
    arr = np.array(fp)
    fp_array = mx.array(arr).astype(mx.float32)
    return fp_array


def get_tanimoto(fps: Optional[mx.array] = None, chunk_size: int = 5000, matrix: bool = False):
    """"
    Calculate Tanimoto similarity score between an mx.array of molecules.
    fps: Optional[mx.array]
        An mx.array of molecular fingerprints.
    chunk_size: int
        The number of chunks to process keep under GPU buffer limits.
    matrix: bool
        Whether to output a matrix or a flattened np.array.
    """
    n = fps.shape[0]
    bits_set = mx.sum(fps, axis=1, keepdims=True)

    results = []

    # process by chunks
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_fps = fps[i:end_i]  # shape: (chunk, bits)
        chunk_bits = bits_set[i:end_i]  # shape: (chunk, 1)

        # computer intersection
        intersections = mx.matmul(chunk_fps, fps.T)

        # Tanimoto calc
        union = chunk_bits + bits_set.T - intersections
        tanimoto_sim = intersections / (union + 1e-7)
        dist_chunk = 1.0 - tanimoto_sim

        if matrix is not True:
            # only keep lower triangle
            for row_idx in range(i, end_i):
                # row in the distance chunk is (row_idx - i)
                actual_row = dist_chunk[row_idx - i, :row_idx]
                results.append(np.array(actual_row))  # move to CPU/NumPy
        else:
            results.append(dist_chunk)

    # convert into numpy array
    return np.concatenate(results)


def butina(fingerprints: mx.array = None, cutoff: float = 0.2, chunk_size: int = 5000) -> list:
    """
    Cluster fingerprints.
    :param fingerprints: mx.array
        A list of RDKit molecular fingerprints.
    :param cutoff: float
        Set the cluster threshold.
    :param chunk_size: int
        The number of chunks to process keep under GPU buffer limits.
    :return:
    """
    # tanimoto matrix
    distance_matrix = get_tanimoto(fingerprints, chunk_size=chunk_size, matrix=False)
    # cluster
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class KMeans:
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', n_init: int = 1, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int = None):
        """
        Initialize the KMeans object. Params should be similar to what can be found on SKlearn:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        :param n_clusters: int
            Number of clusters and centroids to generate.
        :param init: str
            Method of initialization. At this moment, only 'k-means++' is supported.
        :param n_init: int
            Number of times the k-means algorithm is run with different centroid seeds.
        :param max_iter: int
        Maximum number of iterations of the k-means algorithm for a single run.
        :param tol: float
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two
            consecutive iterations to declare convergence.
        :param random_state: int
            Determines random number generation for centroid initialization.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = float('inf')

    def fit(self, X: mx.array):
        """
        Compute k-means clustering.
        :param X: mx.array
            Training instances to cluster. Must be converted to type mx.array().
        :return:
        """
        if not isinstance(X, mx.array):
            X = mx.array(X)
        X = X.astype(mx.float32)

        if self.random_state is not None:
            mx.random.seed(self.random_state)

        N, D = X.shape
        X_sq = mx.sum(X * X, axis=-1, keepdims=True)

        # initialize randomly
        random_indices = mx.random.randint(0, N, [self.n_clusters])
        centers = X[random_indices]

        # pre-create an array of cluster indices [0, 1, 2, ... K-1]
        cluster_indices = mx.arange(self.n_clusters)[None, :]

        for i in range(self.max_iter):
            # distances & assignments
            C_sq = mx.sum(centers * centers, axis=-1)
            distances = X_sq + C_sq - 2.0 * mx.matmul(X, centers.T)
            labels = mx.argmin(distances, axis=-1)

            # one-hot encoded matrix of labels (N, K)
            one_hot = (labels[:, None] == cluster_indices).astype(mx.float32)

            # sum datapoints in each cluster (K, N) matmul (N, D) -> (K, D)
            cluster_sums = mx.matmul(one_hot.T, X)

            # count points in each cluster (K, 1)
            cluster_counts = mx.sum(one_hot, axis=0, keepdims=True).T

            # replace 0 counts with 1 to avoid NaN errors
            safe_counts = mx.maximum(cluster_counts, 1.0)
            new_centers = cluster_sums / safe_counts

            # empty clusters - replace with random data points
            empty_mask = (cluster_counts == 0)
            random_replacements = X[mx.random.randint(0, N, [self.n_clusters])]
            new_centers = mx.where(empty_mask, random_replacements, new_centers)

            # convergence check
            shift = mx.max(mx.sqrt(mx.sum((centers - new_centers) ** 2, axis=-1)))
            centers = new_centers
            self.labels_ = labels

            mx.eval(centers, self.labels_)

            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        return self

    def predict(self, X: mx.array):
        """
        Predict the closest cluster each sample in X belongs to.
        :param X: mx.array
            New data to predict.
        :return:
        """
        X_sq = mx.sum(X * X, axis=-1, keepdims=True)
        C_sq = mx.sum(self.cluster_centers_ * self.cluster_centers_, axis=-1)
        distances = X_sq + C_sq - 2.0 * mx.matmul(X, self.cluster_centers_.T)
        return mx.argmin(distances, axis=-1)

    def pairwise_distances_argmin_min(self, array: mx.array):
        """
        Mimics sklearn.metrics.pairwise_distances_argmin_min.
        :param array: mx.array
            The original data as type mx.array() (shape: N, D)
        """
        if self.cluster_centers_ is None:
            raise ValueError("KMeans Model is not fitted yet!")

        centers = self.cluster_centers_

        # calculate squared distances
        centers_sq = mx.sum(centers * centers, axis=-1, keepdims=True)  # Shape (K, 1)
        array_sq = mx.sum(array * array, axis=-1)  # Shape (N,)

        # distance matrix shape: (K, N)
        distances_sq = centers_sq + array_sq - 2.0 * mx.matmul(centers, array.T)

        # get the index of the minimum distance along the N dimension
        closest_idx = mx.argmin(distances_sq, axis=-1)

        # get the actual minimum distances
        min_sq_distances = mx.min(distances_sq, axis=-1)

        # square root for true Euclidean distance
        min_distances = mx.sqrt(mx.maximum(min_sq_distances, 0.0))

        return closest_idx, min_distances

    def _kmeans_plusplus(self, X, N):
        """
        Support function for k-means++.
        :param X:
        :param N:
        :return:
        """
        centers = []
        first_idx = mx.random.randint(0, N, [1]).item()
        centers.append(X[first_idx])
        X_sq = mx.sum(X * X, axis=-1, keepdims=True)

        for _ in range(1, self.n_clusters):
            current_centers = mx.stack(centers)
            C_sq = mx.sum(current_centers * current_centers, axis=-1)
            distances = X_sq + C_sq - 2.0 * mx.matmul(X, current_centers.T)
            min_distances = mx.min(distances, axis=-1)

            logits = mx.log(min_distances + 1e-8)
            next_idx = mx.random.categorical(logits, shape=[1]).item()
            centers.append(X[next_idx])

        return mx.stack(centers)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
