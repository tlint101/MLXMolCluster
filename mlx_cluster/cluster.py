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


# todo test DBSCAN
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean", chunk_size=5_000):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.chunk_size = chunk_size
        self.labels_ = None

    def fit(self, X: mx.array):
        n_samples = X.shape[0]
        # distance and masking
        dist_matrix = self._compute_distances(X)
        adj_matrix = dist_matrix <= self.eps  # mx.array (bool)

        # core point detection
        neighbor_counts = mx.sum(adj_matrix, axis=1)
        is_core = neighbor_counts >= self.min_samples

        # convert to np.array for calculations
        adj_np = np.array(adj_matrix)
        is_core_np = np.array(is_core)
        self.labels_ = np.full(n_samples, -1)

        cluster_id = 0
        for i in range(n_samples):
            if self.labels_[i] != -1 or not is_core_np[i]:
                continue

            self.labels_[i] = cluster_id
            stack = [i]
            while stack:
                curr = stack.pop()
                # get neighbors usign boolean mask
                neighbors = np.where(adj_np[curr])[0]
                for neighbor in neighbors:
                    if self.labels_[neighbor] == -1:
                        self.labels_[neighbor] = cluster_id
                        if is_core_np[neighbor]:
                            stack.append(neighbor)
            cluster_id += 1
        return self

    def _compute_distances(self, X: mx.array):
        """vectorized distance calculation on mlx."""
        if self.metric == "tanimoto":
            return get_tanimoto(fps=X, chunk_size=self.chunk_size, matrix=True)
        elif self.metric == "euclidean":
            # optimized L2: sqrt(sum(x^2) + sum(y^2) - 2 * x.T * y)
            sq_norms = mx.sum(X ** 2, axis=1, keepdims=True)
            dist_sq = sq_norms + sq_norms.T - 2 * mx.matmul(X, X.T)
            return mx.sqrt(mx.maximum(dist_sq, 0.0))
        elif self.metric == "manhattan" or (self.metric == "minkowski" and self.p == 1):
            # L1 Distance
            return mx.sum(mx.abs(X[:, None, :] - X[None, :, :]), axis=-1)
        elif self.metric == "cosine":
            # cosine Distance = 1 - (A·B / (||A||*||B||))
            norm = mx.sqrt(mx.sum(X ** 2, axis=1, keepdims=True))
            similarity = mx.matmul(X, X.T) / (norm * norm.T + 1e-7)
            return 1.0 - similarity
        else:
            raise ValueError(f"Metric '{self.metric}' is not supported in this MLX implementation.")


# class MLXSpectralClustering:
#     def __init__(self, n_clusters=8, gamma=1.0, affinity='rbf', assign_labels='kmeans'):
#         self.n_clusters = n_clusters
#         self.gamma = gamma
#         self.affinity = affinity
#         self.assign_labels = assign_labels
#         self.labels_ = None
#
#     def fit_predict(self, X):
#         N = X.shape[0]
#
#         # 1. Compute Affinity Matrix (RBF Kernel)
#         # Using the same logic as sklearn: exp(-gamma * ||x-y||^2)
#         sq_norms = mx.sum(X ** 2, axis=1)
#         dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * mx.matmul(X, X.T)
#         A = mx.exp(-self.gamma * dist_sq)
#
#         # 2. Compute Degree Matrix and Laplacian
#         # L = D - A (Unnormalized) or L = I - D^-1/2 A D^-1/2 (Normalized)
#         D = mx.sum(A, axis=1)
#         D_inv_sqrt = 1.0 / mx.sqrt(D)
#         L_norm = mx.eye(N) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
#
#         # 3. Eigen Decomposition
#         # We need the eigenvectors corresponding to the smallest eigenvalues
#         evals, evecs = mx.linalg.eigh(L_norm)
#
#         # 4. Extract Top K Eigenvectors (Spectral Embedding)
#         U = evecs[:, :self.n_clusters]
#
#         # Normalize rows to unit length (important for stability)
#         U = U / mx.linalg.norm(U, axis=1, keepdims=True)
#
#         # 5. Final Step: Run your existing KMeans on the embedding
#         from your_kmeans_file import KMeans  # Use your existing class here
#         km = KMeans(n_clusters=self.n_clusters)
#         self.labels_ = km.fit(U).labels_
#
#         return self.labels_
#
#     class MLXGaussianMixture:
#         def __init__(self, n_components=1, tol=1e-3, max_iter=100, reg_covar=1e-6):
#             self.n_components = n_components
#             self.tol = tol
#             self.max_iter = max_iter
#             self.reg_covar = reg_covar  # Matches sklearn's stability constant
#
#             self.weights_ = None
#             self.means_ = None
#             self.covariances_ = None
#
#         def fit(self, X):
#             N, D = X.shape
#             # Initialize weights uniformly and means randomly from data
#             self.weights_ = mx.full((self.n_components,), 1.0 / self.n_components)
#             self.means_ = X[mx.random.randint(0, N, (self.n_components,))]
#             self.covariances_ = mx.stack([mx.eye(D) for _ in range(self.n_components)])
#
#             prev_log_likelihood = -float('inf')
#
#             for i in range(self.max_iter):
#                 # --- E-Step: Compute Responsibilities ---
#                 resp = self._estimate_responsibilities(X)
#
#                 # --- M-Step: Update Parameters ---
#                 nk = mx.sum(resp, axis=0)  # Total weight in each cluster
#                 self.weights_ = nk / N
#                 self.means_ = mx.matmul(resp.T, X) / nk[:, None]
#
#                 for k in range(self.n_components):
#                     diff = X - self.means_[k]
#                     weighted_diff = diff * mx.sqrt(resp[:, k:k + 1])
#                     # Add regularization to diagonal for numerical stability
#                     self.covariances_[k] = mx.matmul(weighted_diff.T, weighted_diff) / nk[k] + \
#                                            mx.eye(D) * self.reg_covar
#
#                 # Convergence Check
#                 current_log_likelihood = self._compute_log_likelihood(X)
#                 if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
#                     break
#                 prev_log_likelihood = current_log_likelihood
#                 mx.eval(self.means_, self.covariances_)
#
#             return self
#
#         def _estimate_responsibilities(self, X):
#             # Calculates P(cluster | point) using log-sum-exp for stability
#             weighted_log_probs = self._compute_log_prob(X) + mx.log(self.weights_)
#             log_prob_norm = mx.logsumexp(weighted_log_probs, axis=1, keepdims=True)
#             return mx.exp(weighted_log_probs - log_prob_norm)
#
#         def _compute_log_prob(self, X):
#             # Vectorized multivariate normal log-pdf
#             N, D = X.shape
#             probs = []
#             for k in range(self.n_components):
#                 diff = X - self.means_[k]
#                 # MLX handles linalg.inv and det very efficiently on GPU
#                 prec = mx.linalg.inv(self.covariances_[k])
#                 log_det = mx.log(mx.linalg.det(self.covariances_[k]))
#
#                 # Log Mahalanobis distance
#                 dist = mx.sum(mx.matmul(diff, prec) * diff, axis=1)
#                 log_prob = -0.5 * (D * mx.log(2 * 3.14159) + log_det + dist)
#                 probs.append(log_prob)
#             return mx.stack(probs, axis=1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
