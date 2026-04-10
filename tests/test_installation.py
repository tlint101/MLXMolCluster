import pandas as pd
from pathlib import Path
from mlx_cluster import fp_to_mlx, get_tanimoto, butina, KMeans, FPGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "tutorial" / "dataset" / "chembl-33-natural-products-subset.smi"


# script adapted from tutorial to check if it installs and runs
def test():
    # load molecules
    df = pd.read_csv(filepath_or_buffer=DATA_PATH, sep='\t', names=['smiles'], header=None)

    # take first 10_000 molecules
    smi_list = df['smiles'][:10_000].tolist()

    # calculate rdkit fingerprint
    fp_gen = FPGenerator(smi_list)
    rdkit_fps = fp_gen.fingerprint(type='rdkit', nbits=1024, n_cpu=10)

    # convert to mlx
    mlx_fp = fp_to_mlx(rdkit_fps)

    # tanimoto
    get_tanimoto(mlx_fp)

    # run clustering
    butina_mlx = butina(mlx_fp)
    kmeans = KMeans(n_clusters=200, random_state=0).fit(mlx_fp)


if __name__ == "__main__":
    test()
