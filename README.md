This repository contains the machine learning code used in the paper: Exploiting radar polarimetry for nowcasting thunderstorm hazards using deep learning, submitted to Natural Hazards and Earth System Sciences.

# Installation

You need NumPy, Scipy, Matplotlib, Seaborn, Tensorflow (2.6 used in development), Numba, Dask and NetCDF4 for Python.

Clone the repository, then, in the main directory, run
```bash
$ python setup.py develop
```
(if you plan to modify the code) or
```bash
$ python setup.py install
```
if you just want to run it.

# Downloading data

The dataset for the polarimetric variables and quality indices, pretrained models and results can be found at the following Zenodo repository: https://doi.org/10.5281/zenodo.7760740. Follow the instructions there on where to place the data.
The dataset for radar can be found at the following Zenodo repository: https://zenodo.org/record/6802292

# Pretrained models

The pretrained models are available at https://doi.org/10.5281/zenodo.7760740. Unzip the files `models-*.zip` found there to the `runs/run*` directory and `results.zip` to the `runs/run*/results` directory..

# Running

Go to the `scripts` directory and start an interactive shell. There, you can find `training.py` that contains the script you need for training and `plots_sources.py` that produces the plots from the paper.
