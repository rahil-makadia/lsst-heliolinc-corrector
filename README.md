# lsst-heliolinc-corrector

## WARNING: This repository is obsolete. Users should instead use the [GRSS library](https://github.com/rahil-makadia/grss)

Batch least squares differential corrector for orbit determination on clusters generated by heliolinc3D

### Getting started:
To get started, please create a conda environment using the environment.yml file
```
conda env create -f environment.yml
```
This will create a conda environment named "lsst-corrector" that contains all necessary dependencies. The environment can be activated via
```
conda activate lsst-corrector
```

### Run the demo notebook:
In order to run the demo notebook please activate the conda environment named lsst-corrector

```
conda activate lsst-corrector
```
and add it as a Jupyter notebook kernel to be used with the demo .ipynb files on web jupyter sessions
```
python -m ipykernel install --user --name lsst-corrector
```
This last step only needs to be done once.

### Please feel free contact Rahil (makadia2@illinois.edu) with questions/concerns
