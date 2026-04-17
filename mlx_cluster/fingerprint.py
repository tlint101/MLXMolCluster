import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from typing import Optional, Union


class FPGenerator:
    def __init__(self, mols: Optional[Union[list, np.array]] = None):
        """
        Initialize the FPGenerator object. Mols can contain a list or np.array of either molecualr smiles strings or
        ROMol objects.
        :param mols: Optional[Union[list, np.array]]
            A list or np.array of smiles strings or ROMol objects. If a smiles strings, they will be converted into
            ROMols accordingly.
        """
        if isinstance(mols[0], str):  # convert smi str to ROMol
            mol_list = [Chem.MolFromSmiles(mol) for mol in mols]
            self.mols = mol_list
        elif isinstance(mols[0], Chem.rdchem.Mol):  # mols are ROMols
            self.mols = mols
        else:
            self.mols = None

    def fingerprint(self, mols: Optional[Union[list, np.array]] = None, type: str = "morgan", nbits: int = 1024,
                    radius: int = 2, minPath: int = 1, maxPath=7, n_cpu: int = 5):
        """
        Generate a fingerprint for a molecule. There is an option for multithreading. Four types of fingerprints can be
        generated - 'morgan', 'rdkit', 'atompair' or 'AtomPair, and 'torsion'.
        :param mols: Optional[Union[list, np.array]]
            A list or np.array of smiles strings or ROMol objects. If a smiles strings, they will be converted into
            ROMols accordingly. Optional. If given during FPGenerator() initialization, then can be ignored.
        :param type: str
            Set the type of fingerprint. Defaults to 'morgan'.
        :param nbits: int
            Set bit number for a fingerprint. Defaults to 1024.
        :param radius: int
            The number of iterations to grow the fingerprint. Only needed for type 'morgan'.
        :param minPath: int
             The minimum path length (in bonds) to be included. Only needed for type 'rdkit'.
        :param maxPath: int
            The maximum path length (in bonds) to be included. Only needed for type 'rdkit'.
        :param n_cpu: int
            Set the number of CPUs to use. Defaults to 5.
        :return:
        """
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
        else:
            raise ValueError(
                "Type not found! Only 'morgan', 'rdkit', 'atompair' or 'AtomPair', and 'torsion' are supported!")

        return fp
