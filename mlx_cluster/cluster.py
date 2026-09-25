import warnings
import numpy as np
import mlx.core as mx
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from typing import Optional, Union

_VALID_OUTPUTS = ("array", "matrix", "avg", "average")


def fp_to_mlx(fp: Union[list, np.array]) -> mx.array:
    """
    Convert a list of fingerprints into MLX array
    :param fp: Union[list, np.array]
        A list of molecular fingerprints calculated using RDKit, or any array-like of 0/1 values.
    :return:
        An mx.array of shape (n_fingerprints, nbits) and dtype float32.
    """
    # RDKit bit vectors take a packed-bytes fast path: np.array() on a list of ExplicitBitVect falls back to
    # NumPy's Python sequence protocol and costs one __getitem__ per bit (seconds for 10k x 1024).
    if len(fp) and isinstance(fp[0], DataStructs.ExplicitBitVect):
        nbits = fp[0].GetNumBits()
        packed = np.frombuffer(b"".join(DataStructs.BitVectToBinaryText(f) for f in fp), dtype=np.uint8)
        # BitVectToBinaryText emits ceil(nbits/8) bytes, unpacking fingerprint whose length is not a multiple of 8
        # yields trailing pad bits that must be trimmed.
        arr = np.unpackbits(packed.reshape(len(fp), -1), axis=1, bitorder="little")[:, :nbits]
    else:
        arr = np.asarray(fp)
    fp_array = mx.array(arr).astype(mx.float32)
    return fp_array


def get_tanimoto(fps: mx.array, chunk_size: int = 5000, output: str = "array"):
    """
    Calculate Tanimoto similarity score between an mx.array of molecules.

    With binary fingerprints every partial sum of the intersection matmul is an exact integer, and float32 holds
    integers exactly up to 2**24, so the intersection and union counts carry no rounding error for any realistic
    nbits. Only the final division rounds.

    :param fps: mx.array
        An mx.array of molecular fingerprints. Assumed to hold 0/1 values.
    :param chunk_size: int
        The number of rows processed at a time, to keep under GPU buffer limits.
    :param output: str
        Determine the type of output given. Only three str type can be used: "array" will output the results as a
        flattened array, "matrix" will output the results as an np.array aligned in a matrix, and "avg" or "average"
        will output the average Tanimoto Similarity score for the given dataset.
    :return:
        For "array", an np.array of shape (N*(N-1)/2,) holding the lower triangle of the distance matrix. For
        "matrix", an (N, N) np.array of distances. For "avg" or "average", a float giving the mean pairwise
        Tanimoto similarity.
    """
    if output not in _VALID_OUTPUTS:
        raise ValueError("Unknown output type! Can only be 'array', 'matrix', or 'average' or 'avg'!")

    n = fps.shape[0]
    # input check
    if n < 2:
        raise ValueError("Need at least 2 fingerprints needed to calculate Tanimoto similarity!")

    bits_set = mx.sum(fps, axis=1, keepdims=True)

    # "array"/"avg" read lower triangle only (skips half matmul); "matrix" needs all columns.
    full_width = output == "matrix"

    # hold outputs: preallocate to avoid transiently doubling memory via np.concatenate
    total_sim_sum = 0.0
    results = None
    if output == "array":
        results = np.empty(n * (n - 1) // 2, dtype=np.float32)
    elif output == "matrix":
        results = np.empty((n, n), dtype=np.float32)

    # process by chunks
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        width = n if full_width else end_i
        chunk_fps = fps[i:end_i]  # shape: (chunk, bits)
        chunk_bits = bits_set[i:end_i]  # shape: (chunk, 1)

        # computer intersection
        intersections = mx.matmul(chunk_fps, fps[:width].T)

        # Tanimoto calc
        union = chunk_bits + bits_set[:width].T - intersections
        tanimoto_sim = intersections / (union + 1e-7)

        # one eval per chunk; use zero-copy NumPy views. Keep mx.array referenced to prevent memory corruption.
        if output == "array":
            dist_chunk = 1.0 - tanimoto_sim
            mx.eval(dist_chunk)
            view = np.asarray(memoryview(dist_chunk))
            base = i * (i - 1) // 2  # row r occupies [r(r-1)/2, r(r+1)/2)
            for row_idx in range(i, end_i):
                results[base:base + row_idx] = view[row_idx - i, :row_idx]
                base += row_idx
        elif output == "matrix":
            dist_chunk = 1.0 - tanimoto_sim
            mx.eval(dist_chunk)
            results[i:end_i] = np.asarray(memoryview(dist_chunk))
        else:
            mx.eval(tanimoto_sim)
            view = np.asarray(memoryview(tanimoto_sim))
            for row_idx in range(i, end_i):
                # Accumulate float64 to avoid frequent GPU syncs
                total_sim_sum += float(view[row_idx - i, :row_idx].sum(dtype=np.float64))

    # final outputs
    if output == "array" or output == "matrix":
        return results
    total_pairs = (n * (n - 1)) / 2
    avg_sim = total_sim_sum / total_pairs
    return avg_sim


def butina(fingerprints: mx.array, cutoff: float = 0.2, chunk_size: int = 5000) -> list:
    """
    Cluster fingerprints.
    :param fingerprints: mx.array
        A list of RDKit molecular fingerprints.
    :param cutoff: float
        Set the cluster threshold.
    :param chunk_size: int
        The number of rows processed at a time, to keep under GPU buffer limits.
    :return:
        A list of clusters, largest first. Each cluster is a tuple of molecule indices whose
        first element is the cluster centroid.
    """
    # tanimoto matrix
    distance_matrix = get_tanimoto(fingerprints, chunk_size=chunk_size, output='matrix')
    # cluster
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def _split_key(key):
    """
    Derive a fresh subkey from an explicit MLX RNG key, passing ``None`` straight through.
    :param key: Optional[mx.array]
        A key from ``mx.random.key``, or None to draw from MLX's global RNG stream.
    :return:
        A (key_to_use, next_key) tuple; both are None when ``key`` is None.
    """
    if key is None:
        return None, None
    use, nxt = mx.random.split(key)
    return use, nxt


class KMeans:
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', n_init: int = 1, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int = None):
        """
        Initialize the KMeans object. Params should be similar to what can be found on SKlearn:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        :param n_clusters: int
            Number of clusters/centroids.
        :param init: str
            Initialization method ('k-means++' or 'random').
        :param n_init: int
            Number of runs with different seeds; best result is kept.
        :param max_iter: int
            Maximum iterations per run.
        :param tol: float
            Convergence tolerance (Frobenius norm).
        :param random_state: int
            Set random state.
        """
        if init not in ('k-means++', 'random'):
            raise ValueError("init must be either 'k-means++' or 'random'!")
        if n_init < 1:
            raise ValueError("n_init must be at least 1!")
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
        Compute k-means clustering. Runs ``n_init`` independent initializations and keeps the one with the
        lowest inertia.
        :param X: mx.array
            Training instances to cluster. Must be converted to type mx.array().
        :return:
            self
        """
        if not isinstance(X, mx.array):
            X = mx.array(X)
        X = X.astype(mx.float32)

        N, D = X.shape
        if self.n_clusters > N:
            raise ValueError(f"n_clusters={self.n_clusters} cannot exceed the number of samples ({N})!")

        # an explicit key keeps the seeding local; mx.random.seed() would reseed the caller's global stream.
        key = mx.random.key(self.random_state) if self.random_state is not None else None

        X_sq = mx.sum(X * X, axis=-1, keepdims=True)

        best = None
        for _ in range(self.n_init):
            use, key = _split_key(key)
            centers, labels, inertia = self._single_run(X, X_sq, use)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)

        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def _single_run(self, X: mx.array, X_sq: mx.array, key):
        """
        Run Lloyd's algorithm once from a single initialization.
        :param X: mx.array
            Training instances, float32.
        :param X_sq: mx.array
            Precomputed row-wise squared norms of X, shape (N, 1).
        :param key: Optional[mx.array]
            MLX RNG key, or None to use the global stream.
        :return:
            A (centers, labels, inertia) tuple.
        """
        N, D = X.shape

        use, key = _split_key(key)
        centers = self._init_centers(X, X_sq, use)

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
            use, key = _split_key(key)
            random_replacements = X[mx.random.randint(0, N, [self.n_clusters], key=use)]
            new_centers = mx.where(empty_mask, random_replacements, new_centers)

            # convergence check
            shift = mx.max(mx.sqrt(mx.sum((centers - new_centers) ** 2, axis=-1)))
            centers = new_centers

            mx.eval(centers)

            if shift < self.tol:
                break

        # use final centers for labels, in-loop labels used previous iteration's centroids.
        C_sq = mx.sum(centers * centers, axis=-1)
        distances = X_sq + C_sq - 2.0 * mx.matmul(X, centers.T)
        labels = mx.argmin(distances, axis=-1)
        # clamped because the expanded ||x||^2 + ||c||^2 - 2x.c form can go slightly negative on binary data
        inertia = float(mx.sum(mx.maximum(mx.min(distances, axis=-1), 0.0)).item())
        mx.eval(centers, labels)

        return centers, labels, inertia

    def _init_centers(self, X: mx.array, X_sq: mx.array, key):
        """
        Choose the initial centroids according to ``self.init``.
        :param X: mx.array
            Training instances, float32.
        :param X_sq: mx.array
            Precomputed row-wise squared norms of X, shape (N, 1).
        :param key: Optional[mx.array]
            MLX RNG key, or None to use the global stream.
        :return:
            An mx.array of shape (n_clusters, D).
        """
        if self.init == 'k-means++':
            return self._kmeans_plusplus(X, X_sq, key)
        # sampled without replacement, so the same point cannot seed two clusters
        indices = mx.random.permutation(X.shape[0], key=key)[:self.n_clusters]
        return X[indices]

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

    def _kmeans_plusplus(self, X: mx.array, X_sq: mx.array, key):
        """
        Support function for k-means++ (D^2 sampling). Uses running nearest-center distances to achieve O(K*N*D)
        complexity (vs O(K^2*N*D)) by only checking against the newest center. 0-D MLX indexing keeps loops on GPU.
        :param X: mx.array
            Training instances (float32).
        :param X_sq: mx.array
            Precomputed row-wise squared norms (N, 1).
        :param key: Optional[mx.array]
            MLX RNG key; defaults to global stream.
        :return: mx.array
            Shape (n_clusters, D).
        """
        N = X.shape[0]
        # flat (N,) norms; the expanded form below keeps each step a matrix-vector product instead of
        # materializing an (N, D) difference tensor per cluster
        x_norms = X_sq.reshape(-1)

        use, key = _split_key(key)
        center = X[mx.random.randint(0, N, [1], key=use)[0]]
        centers = [center]

        min_distances = mx.maximum(x_norms + mx.sum(center * center) - 2.0 * mx.matmul(X, center), 0.0)

        for _ in range(1, self.n_clusters):
            # p ∝ squared distance to the nearest chosen centre; the floor keeps duplicate fingerprints, which
            # give an exact distance of 0, from putting log(0) = -inf into the logits
            logits = mx.log(mx.maximum(min_distances, 1e-8))
            use, key = _split_key(key)
            next_idx = mx.random.categorical(logits, key=use)
            center = X[next_idx]
            centers.append(center)

            distances = mx.maximum(x_norms + mx.sum(center * center) - 2.0 * mx.matmul(X, center), 0.0)
            min_distances = mx.minimum(min_distances, distances)
            mx.eval(min_distances, center)

        return mx.stack(centers)


class HDBSCAN:
    """
    Hierarchical density-based clustering, in the spirit of sklearn's ``HDBSCAN``.

    Builds a cluster hierarchy across all density levels and keeps the clusters that persist longest, so no
    ``eps`` parameter is needed. Unlike :func:`butina`, which assigns every molecule, HDBSCAN labels points in
    sparse regions as noise (``-1``).

    Distances and core distances are computed on the GPU; the minimum spanning tree, hierarchy condensation and
    cluster extraction run on the CPU.

    If you compare against sklearn yourself, pass ``copy=True``: with ``metric='precomputed'`` sklearn defaults
    to ``copy=False`` and modifies your distance matrix in place, so reusing one matrix across several calls
    silently corrupts every result after the first.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int = None, metric: str = 'tanimoto',
                 chunk_size: int = 5000):
        """
        Initialize the HDBSCAN object.
        :param min_cluster_size: int
            The smallest group of molecules that counts as a cluster. Anything smaller becomes noise.
        :param min_samples: int
            Neighbor rank used for the core distance; higher values make the result more conservative and
            produce more noise. Defaults to ``min_cluster_size``.
        :param metric: str
            Distance metric, either 'tanimoto' (for fingerprints) or 'euclidean'.
        :param chunk_size: int
            The number of rows processed at a time, to keep under GPU buffer limits.
        """
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2!")
        if min_samples is not None and min_samples < 1:
            raise ValueError("min_samples must be at least 1!")
        if metric not in ('tanimoto', 'euclidean'):
            raise ValueError("Metric must be either 'tanimoto' or 'euclidean'!")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.chunk_size = chunk_size
        self.labels_ = None

    def fit(self, X: mx.array):
        """
        Compute the clustering.
        :param X: mx.array
            Molecular fingerprints, shape (N, nbits). For metric='tanimoto' these must hold 0/1 values.
        :return:
            self
        """
        if not isinstance(X, mx.array):
            X = mx.array(X)
        X = X.astype(mx.float32)

        n = X.shape[0]
        min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
        if n < max(self.min_cluster_size, min_samples):
            raise ValueError(f"Need at least {max(self.min_cluster_size, min_samples)} samples to cluster!")

        distances = self._compute_distances(X)
        core = self._core_distances(distances, min_samples)     # on GPU, from the float32 matrix
        distances = distances.astype(np.float64)                # float64 makes the CPU MST below faster
        linkage = self._single_linkage(self._mutual_reachability_mst(distances, core), n)
        tree = self._condense_tree(linkage, n, self.min_cluster_size)
        self.labels_ = self._extract_clusters(tree, self._cluster_stability(tree), n)
        return self

    def fit_predict(self, X: mx.array) -> np.ndarray:
        """
        Compute the clustering and return the labels.
        :param X: mx.array
            Molecular fingerprints, shape (N, nbits).
        :return:
            A (N,) np.array of cluster labels, with -1 marking noise.
        """
        return self.fit(X).labels_

    def _compute_distances(self, X: mx.array) -> np.ndarray:
        """
        Build the full distance matrix on the GPU.
        :param X: mx.array
            Molecular fingerprints, shape (N, nbits).
        :return:
            A square (N, N) np.array of distances.
        """
        if self.metric == 'tanimoto':
            return get_tanimoto(X, chunk_size=self.chunk_size, output='matrix')
        squared = mx.sum(X * X, axis=1, keepdims=True)
        distances = mx.sqrt(mx.maximum(squared + squared.T - 2.0 * mx.matmul(X, X.T), 0.0))
        mx.eval(distances)
        return np.asarray(memoryview(distances)).copy()

    @staticmethod
    def _core_distances(distances: np.ndarray, min_samples: int) -> np.ndarray:
        """
        Find the distance from each point to its ``min_samples``-th nearest neighbour, the point itself
        counting as the first.
        :param distances: np.ndarray
            Square (N, N) float32 distance matrix.
        :param min_samples: int
            Neighbour rank defining the core distance.
        :return:
            A (N,) float64 array of core distances.
        """
        ranked = mx.partition(mx.array(distances), min_samples - 1, axis=1)[:, min_samples - 1]
        mx.eval(ranked)
        return np.asarray(memoryview(ranked)).astype(np.float64)

    @staticmethod
    def _mutual_reachability_mst(distances: np.ndarray, core: np.ndarray) -> list:
        """
        Build a minimum spanning tree over the mutual reachability graph with Prim's algorithm.
        :param distances: np.ndarray
            Square (N, N) distance matrix.
        :param core: np.ndarray
            Core distances, shape (N,).
        :return:
            A list of (node_a, node_b, weight) edges, N-1 of them.
        """
        n = distances.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf)
        parent = np.full(n, -1, dtype=np.int64)
        best[0] = 0.0
        edges = []
        for _ in range(n):
            current = int(np.argmin(np.where(in_tree, np.inf, best)))
            in_tree[current] = True
            if parent[current] >= 0:
                edges.append((int(parent[current]), current, float(best[current])))
            reach = np.maximum(np.maximum(distances[current], core[current]), core)
            better = (~in_tree) & (reach < best)
            best[better] = reach[better]
            parent[better] = current
        return edges

    @staticmethod
    def _single_linkage(edges: list, n: int) -> np.ndarray:
        """
        Build a SciPy-style linkage matrix from MST edges by union-find.
        :param edges: list
            (node_a, node_b, weight) tuples from the MST.
        :param n: int
            Number of original points.
        :return:
            An (N-1, 4) array of (left, right, distance, size); internal nodes are numbered N, N+1, ...
        """
        edges = sorted(edges, key=lambda e: e[2])
        parent = np.arange(2 * n - 1)
        size = np.ones(2 * n - 1, dtype=np.int64)
        current = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        linkage = np.zeros((n - 1, 4))
        for k, (a, b, weight) in enumerate(edges):
            ra, rb = find(current[a]), find(current[b])
            new_node = n + k
            linkage[k] = (ra, rb, weight, size[ra] + size[rb])
            parent[ra] = parent[rb] = new_node
            size[new_node] = size[ra] + size[rb]
            current[a] = current[b] = new_node
        return linkage

    @staticmethod
    def _subtree_points(linkage: np.ndarray, n: int, node: int) -> list:
        """
        Collect every original point beneath a linkage node.
        :param linkage: np.ndarray
            Linkage matrix from :meth:`_single_linkage`.
        :param n: int
            Number of original points.
        :param node: int
            Node id to descend from.
        :return:
            A list of point indices.
        """
        if node < n:
            return [node]
        stack, points = [node], []
        while stack:
            x = stack.pop()
            if x < n:
                points.append(x)
            else:
                stack.append(int(linkage[x - n, 0]))
                stack.append(int(linkage[x - n, 1]))
        return points

    @staticmethod
    def _condense_tree(linkage: np.ndarray, n: int, min_cluster_size: int) -> np.ndarray:
        """
        Condense the dendrogram: a child smaller than ``min_cluster_size`` falls out of its parent rather than
        counting as a genuine split.
        :param linkage: np.ndarray
            Linkage matrix from :meth:`_single_linkage`.
        :param n: int
            Number of original points.
        :param min_cluster_size: int
            Smallest group that counts as a cluster.
        :return:
            A structured array with fields parent, child, lambda_val, child_size.
        """
        root = 2 * n - 2
        relabel = np.zeros(2 * n - 1, dtype=np.int64)
        relabel[root] = n
        next_label = n + 1
        ignore = np.zeros(2 * n - 1, dtype=bool)
        rows = []

        # Descending node id is a valid topological order: in a linkage matrix a node's children always have
        # smaller ids than the node itself, so every parent is relabelled before its children are reached.
        for node in range(root, n - 1, -1):
            if ignore[node]:
                continue
            left, right = int(linkage[node - n, 0]), int(linkage[node - n, 1])
            distance = linkage[node - n, 2]
            lambda_val = np.inf if distance <= 0 else 1.0 / distance
            left_size = 1 if left < n else int(linkage[left - n, 3])
            right_size = 1 if right < n else int(linkage[right - n, 3])
            left_big, right_big = left_size >= min_cluster_size, right_size >= min_cluster_size

            if left_big and right_big:
                for child, child_size in ((left, left_size), (right, right_size)):
                    relabel[child] = next_label
                    rows.append((relabel[node], next_label, lambda_val, child_size))
                    next_label += 1
                continue

            if left_big or right_big:
                keep, drop = (left, right) if left_big else (right, left)
                relabel[keep] = relabel[node]
                falling = [drop]
            else:
                falling = [left, right]

            for child in falling:
                for point in HDBSCAN._subtree_points(linkage, n, child):
                    rows.append((relabel[node], point, lambda_val, 1))
                stack = [child]
                while stack:
                    x = stack.pop()
                    ignore[x] = True
                    if x >= n:
                        stack.append(int(linkage[x - n, 0]))
                        stack.append(int(linkage[x - n, 1]))

        return np.array(rows, dtype=[('parent', np.int64), ('child', np.int64),
                                     ('lambda_val', float), ('child_size', np.int64)])

    @staticmethod
    def _cluster_stability(tree: np.ndarray) -> dict:
        """
        Measure the stability of each candidate cluster: how long its members persist past the cluster's own
        birth.
        :param tree: np.ndarray
            Condensed tree from :meth:`_condense_tree`.
        :return:
            A dict mapping cluster id to stability.
        """
        births = {child: lambda_val for _, child, lambda_val, size in tree if size > 1}
        clusters = sorted(set(tree['parent'].tolist()))
        if clusters:
            births[min(clusters)] = 0.0
        stability = {}
        for cluster in clusters:
            rows = tree['parent'] == cluster
            birth = births.get(cluster, 0.0)
            stability[cluster] = float(np.sum((tree['lambda_val'][rows] - birth) * tree['child_size'][rows]))
        return stability

    @staticmethod
    def _extract_clusters(tree: np.ndarray, stability: dict, n: int) -> np.ndarray:
        """
        Select the final clusters by excess of mass: keep a cluster when it is more stable than its descendants
        combined.
        :param tree: np.ndarray
            Condensed tree from :meth:`_condense_tree`.
        :param stability: dict
            Cluster stabilities from :meth:`_cluster_stability`.
        :param n: int
            Number of original points.
        :return:
            A (N,) array of labels, -1 for noise.
        """
        labels = np.full(n, -1, dtype=np.int64)
        if not stability:
            return labels

        cluster_tree = tree[tree['child_size'] > 1]
        nodes = sorted(stability.keys(), reverse=True)
        root = min(nodes)
        selected = {node: True for node in nodes}

        for node in nodes:
            if node == root:
                continue
            children = cluster_tree['child'][cluster_tree['parent'] == node]
            below = float(sum(stability[c] for c in children))
            if below > stability[node]:
                selected[node] = False
                stability[node] = below
            else:
                stack = children.tolist()
                while stack:
                    x = stack.pop()
                    selected[x] = False
                    stack.extend(cluster_tree['child'][cluster_tree['parent'] == x].tolist())
        selected[root] = False

        for label, cluster in enumerate(sorted(c for c in selected if selected[c])):
            stack, points = [cluster], []
            while stack:
                x = stack.pop()
                rows = tree['parent'] == x
                for child, child_size in zip(tree['child'][rows], tree['child_size'][rows]):
                    if child_size == 1:
                        points.append(int(child))
                    else:
                        stack.append(int(child))
            labels[points] = label
        return labels


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
