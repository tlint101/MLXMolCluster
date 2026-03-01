import numpy as np
import mlx.core as mx
from rdkit.ML.Cluster import Butina
from typing import Optional
import rdkit


def fp_to_mlx(fp: list):
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


class Cluster:
    def __init__(self):
        pass

    def get_tanimoto(self, fps: Optional[mx.array] = None, chunk_size: int = 5000, matrix: bool = False):
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

    def butina(self, fingerprints: list[rdkit.DataStructs.cDataStructs.ExplicitBitVect] = None, cutoff:float=0.2,
               chunk_size:int=5000):
        """
        Cluster fingerprints.
        :param fingerprints: list[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
            A list of RDKit molecular fingerprints.
        :param cutoff: float
            Set the cluster threshold.
        :param chunk_size: int
            The number of chunks to process keep under GPU buffer limits.
        :return:
        """
        # tanimoto matrix
        distance_matrix = self.get_tanimoto(fingerprints, chunk_size=chunk_size, matrix=False)
        # cluster
        clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters
