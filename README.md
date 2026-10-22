# OTProf: estimating high-resolution profiles of optical turbulence ($C_n^2$) from reanalysis using deep learning

Code accompanying the publication

> Pierzyna, Maximilian, et al. “OTProf: Estimating High-Resolution Profiles of Optical Turbulence ($C_n^2$) from Reanalysis Using Deep Learning.” arXiv:2604.09346, arXiv, 10 Apr. 2026. arXiv.org, https://doi.org/10.48550/arXiv.2604.09346.

The repository covers the whole chain from the WRF simulations to the figures of the paper:

1. **WRF simulations** ([`data`](data)). The [`wrf-massive`](https://github.com/mpierzyna/wrf-massive) package orchestrates the WRF runs that generate the high-resolution training data over the Netherlands ([`data/nl/wrf`](data/nl/wrf), 2017-2020) and the out-of-domain test data over the Canary Islands ([`data/canaries/wrf`](data/canaries/wrf), 2017). The repository root is the uv project of these campaigns.
2. **Dataset, model and paper figures** ([`code`](code), its own uv project). A DVC pipeline turns the post-processed WRF output and ERA5 into the training and evaluation datasets; the height-aware Squeezeformer (SQF-H) is trained on WRF interpolated to ERA5 pressure levels and applied to ERA5; the paper figures are generated from its predictions. See [`code/README.md`](code/README.md).

The processed datasets, the trained models and the predictions are available on [Zenodo](https://zenodo.org/records/20733692).
