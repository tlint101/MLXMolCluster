import pandas as pd
from pathlib import Path
from mlx_cluster import fp_to_mlx, get_tanimoto, butina, KMeans, FPGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "tutorial" / "dataset" / "chembl-33-natural-products-subset.smi"

N_MOLS = 10_000
N_BITS = 1024
N_CLUSTERS = 200


# script adapted from tutorial to check if it installs and runs
def test():
    # load molecules
    df = pd.read_csv(filepath_or_buffer=DATA_PATH, sep='\t', names=['smiles'], header=None)

    # take first 10_000 molecules
    smi_list = df['smiles'][:N_MOLS].tolist()

    # calculate rdkit fingerprint
    fp_gen = FPGenerator(smi_list)
    rdkit_fps = fp_gen.fingerprint(type='rdkit', nbits=N_BITS, n_cpu=10)
    assert len(rdkit_fps) == N_MOLS

    # convert to mlx
    mlx_fp = fp_to_mlx(rdkit_fps)
    assert mlx_fp.shape == (N_MOLS, N_BITS)

    # tanimoto - flattened lower triangle of the distance matrix
    dists = get_tanimoto(mlx_fp)
    assert dists.shape == (N_MOLS * (N_MOLS - 1) // 2,)
    assert dists.min() >= 0.0 and dists.max() <= 1.0

    # run clustering - every molecule lands in exactly one cluster
    butina_mlx = butina(mlx_fp)
    assert sum(len(cluster) for cluster in butina_mlx) == N_MOLS
    assert len({idx for cluster in butina_mlx for idx in cluster}) == N_MOLS

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(mlx_fp)
    assert kmeans.cluster_centers_.shape == (N_CLUSTERS, N_BITS)
    assert kmeans.labels_.shape == (N_MOLS,)
    assert kmeans.labels_.min() >= 0 and kmeans.labels_.max() < N_CLUSTERS


if __name__ == "__main__":
    test()
