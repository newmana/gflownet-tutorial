# GFlowNet Tutorial

## Prerequisites
* Conda and Mamba environment:
  * ```$ conda install -c conda-forge mamba```
  * https://github.com/conda-forge/miniforge

### Remove Existing Environment
```conda remove --name gflownet-tutorial --all```

### Create Conda Environment

```conda create --name gflownet-tutorial python=3.12 jupyter```
```conda activate gflownet-tutorial```
```python -m pip install matplotlib numpy torch torchvision tqdm```
