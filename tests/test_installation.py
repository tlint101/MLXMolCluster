import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from mcluster import fp_to_mlx, get_tanimoto, butina, KMeans


# script adapted from tutorial to check if it installs and runs
def test():
    # load molecules
    df = pd.read_csv(filepath_or_buffer="../tutorial/dataset/chembl-33-natural-products-subset.smi", sep='\t',
                     names=['smiles'], header=None)

    # take first 10_000 molecules
    smi_list = df['smiles'][:10_000].tolist()

    # convert smi into RDKit object
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]

    # set rdkit fingerprint generator
    fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    # calculate rdkit fingerprint
    rdkit_fps = [fp_gen.GetFingerprint(mol) for mol in mol_list]

    mlx_fp = fp_to_mlx(rdkit_fps)

    # tanimoto
    get_tanimoto(mlx_fp)

    # run clustering
    butina_mlx = butina(mlx_fp)
    kmeans = KMeans(n_clusters=200, random_state=0).fit(mlx_fp)


if __name__ == "__main__":
    test()
