[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://github.com/tlint101/MLXMolCluster)
[![MLX](https://img.shields.io/badge/MLX-0.30.0+-black?logo=apple&labelColor=gray)](https://github.com/ml-explore/mlx)

# MLXMolCluster

Leverage Apple Silicon to cluster molecules using MLX. 

At the time of writing, the project contains two clustering methods:
- Butina
- KMeans

Additional clustering methods will be added over time.

Examples have been written and can be found [here](tutorial). 

## Installation
Clone and install locally:
```python
pip install https://github.com/tlint101/MLXMolCluster.git
```

## Example
The following is an example of clustering molecules using Butina on MLX.
```python
# generate molecular fingerprints
fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
rdkit_fps = [fp_gen.GetFingerprint(mol) for mol in mol_list]

# convert to mlx arrays
mlx_fp = fp_to_mlx(rdkit_fps)

# Butina cluster
butina_mlx = butina(mlx_fp)
```

A speed comparison can be seen at the [tutorial section](tutorial). The runs were performed on a M2 Pro chip 
(10 CPU, 16 GPU)
![Comparisons of Clustering Methods](img/cluster_compare.png "Clustering 10,000 Molecules")

**NOTE:** The figure can be misleading. The figure shows the clustering speed of already generated molecular 
fingerprints. The main bottleneck of clustering remains on the generation of molecular fingerprints. This is done on the 
CPU before being converted to the GPU. Depending on the number of molecules, this can be time intensive. 