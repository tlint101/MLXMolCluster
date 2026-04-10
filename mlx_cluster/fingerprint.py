import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from typing import Optional, Union


class FPGenerator:
    def __init__(self, mols: Optional[Union[list, np.array]] = None):
        if isinstance(mols[0], str):  # convert smi str to ROMol
            mol_list = [Chem.MolFromSmiles(mol) for mol in mols]
            self.mols = mol_list
        elif isinstance(mols[0], Chem.rdchem.Mol):  # mols are ROMols
            self.mols = mols
        else:
            self.mols = None

    def fingerprint(self, mols: Optional[Union[list, np.array]] = None, type: str = "morgan", nbits: int = 1024,
                    radius: int = 2, minPath: int = 1, maxPath=7, n_cpu: int = 5):
        global fp
        if mols is None:
            mols = self.mols
        if mols is None and self.mols is None:
            raise ValueError("A list or np.array of RDMols must be given!")

        if type == "morgan":
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
            fp = fp_gen.GetFingerprints(mols, numThreads=n_cpu)
        elif type == "rdkit":
            fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=minPath, maxPath=maxPath, fpSize=nbits)
            fp = fp_gen.GetFingerprints(mols, numThreads=n_cpu)
        elif type == "atompair" or type == 'AtomPair':
            fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)
            fp = fp_gen.GetFingerprints(mols, numThreads=n_cpu)
        elif type == "torsion":
            fp_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits)
            fp = fp_gen.GetFingerprints(mols, numThreads=n_cpu)

        return fp
