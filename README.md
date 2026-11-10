<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# OTProf-WRF

Simulate horizontal and vertical profiles of atmospheric (optical) turbulence with the
[Weather Research and Forecasting (WRF) model](https://github.com/wrf-model/wrf). The simulations are orchestrated by [`wrf-massive`](https://github.com/mpierzyna/wrf-massive),
which allows to parameterize WRF simulations and their entire pipeline (forcing download, WPS, 
WRF, Cn2 postprocessing).

The domains [`nl`](./domains/nl/) and [`canaries`](./domains/canaries/) are used in

> Pierzyna, Maximilian, et al. “OTProf: Estimating High-Resolution Profiles of Optical Turbulence ($C_n^2$) from Reanalysis Using Deep Learning.” arXiv:2604.09346, arXiv, 10 Apr. 2026. arXiv.org, https://doi.org/10.48550/arXiv.2604.09346.

This repository is part of a set of repositories:
- [`otprof`](https://github.com/mpierzyna/otprof): Code to train a new OTProf model or run a pretrained model.
- [`otprof-wrf`](https://github.com/mpierzyna/otprof-wrf) (**this repository**): Code to generate the WRF-based $C_n^2$ training data. The generated training dataset and a snapshot of this code are available on Zenodo (DOI: 10.5281/zenodo.23733024).
- [`otprof-ui`](https://github.com/mpierzyna/otprof-ui): Code of the web UI to show OTProf-based $C_n^2$ forecasts and climatology. Check it out at [**https://otprof.mpier.eu**](https://otprof.mpier.eu)!

Please refer to the [License](#license) and [How to cite](#how-to-cite) sections if you would like to use this software or its products for your own work.

This software has been developed by [**Maximilian Pierzyna**](https://orcid.org/0000-0002-0596-8374), Technische Universiteit Delft.


## Setup

1. Clone with the [`wrf-massive`](https://github.com/mpierzyna/wrf-massive) submodule:
   ```bash
   git clone --recurse-submodules git@github.com:mpierzyna/otprof-wrf.git
   # or, in an existing clone
   git submodule update --init submodules/wrf-massive
   ```
2. Create the Python environment (Python 3.12, [uv](https://docs.astral.sh/uv/)). `wrf-massive` is installed
   editable; its `wrf-python` dependency needs a working `gfortran`.
   ```bash
   uv sync
   ```
3. Provide compiled WPS and WRF (v4.5) and the WPS geographical data. Either use an existing installation or build
   the copies that `wrf-massive` carries as optional submodules (`submodules/wrf-massive/submodules/{WPS,WRF}`).
   Their paths are set per machine in `domains/<domain>/env_<machine>.yaml`.
4. Access to the forcing: the CERRA (and ERA5) grib files are pulled with `rclone` from the TU Delft mirror, so an
   `rclone` remote named `tudelft` is needed.

## Usage

Each domain is a self-contained `wrf-massive` workspace; run everything from its directory. See the domain READMEs
for the pipelines and machines.

```bash
cd domains/nl
ln -sf env_<machine>.yaml env.yaml                       # selects the machine paths and the pipeline

uv run python cli.py init-sims simulations.py sims_1y   # (re)create the sim_*/ dirs and simulation.yaml
uv run python cli.py run --stages=wps,wrf sim_2017-01-01 # run stages locally (all stages if omitted)
uv run python cli.py submit -s wrf -j slurm_<machine>.job.sh sim_2017-01-01   # or submit them to SLURM
```

Stage names are those of the pipeline selected in `env.yaml` (`pipeline.py`), e.g. for the full `nl` pipeline
`cerra`, `wps`, `cerra_gc`, `wrf`, `cn2` (Cn2 post-processing) and `sim_done`. Finished stages are marked with
`.done` files and skipped on re-runs. The final output of a simulation is the compressed NetCDF in `sim_*/4_postproc/`.

## AI statement
Claude Opus 5.5 (Anthropic, model ID `claude-opus-5-5`), used through Claude Code, assisted with the development of the code, documentation, and project configuration in this repository. The author has carefully reviewed all AI-assisted contributions.

## License
Unless otherwise noted, the content of this repository is licensed under **GNU General Public License v3.0 or later** (see [`COPYING`](./COPYING)). The repository follows the [REUSE](https://reuse.software) specification: every file states its license, in its header or in [`REUSE.toml`](./REUSE.toml); check with `reuse lint`.

WRF and WPS are not part of this repository. The training dataset generated with this code is licensed separately, under CC-BY-SA-4.0 (see [How to cite](#how-to-cite)).

### Third-party code
`Vtable.CERRA`, `setup_wps.sh` and `run_wps.sh` in `domains/*/sim_*/2_wps/` are derived from [EU_SCORES_project](https://github.com/HarishBaki/EU_SCORES_project) (`WRFV4.4/Vtable.CERRA` and `WRFV4.4/WPS_pipeline.sh`), Copyright (c) 2024 Harish Baki, licensed under the MIT License. These files remain under the MIT License (see [`LICENSES/MIT.txt`](LICENSES/MIT.txt)); they were modified by Maximilian Pierzyna.

## Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the program "OTProf-WRF" written by the Author(s). Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2026, Maximilian Pierzyna (excluding third-party code listed above)

## How to cite
When using OTProf-WRF, please cite the **related article** and the **training dataset** with the code snapshot. You can do so either using the `./CITATION.cff` file or by using the following information:

### Article

> Pierzyna, M., Basu, S., & Saathof, R. (2026). OTProf: estimating high-resolution profiles of optical turbulence ($C_n^2$) from reanalysis using deep learning. *arXiv preprint* arXiv:2604.09346. https://doi.org/10.48550/arXiv.2604.09346

### Training dataset

Our training dataset is licensed under CC-BY-SA-4.0, so you may reuse it for your own work given that you attribute the original author as follows.

> Pierzyna, M. (2026). *OTProf: Estimating High-Resolution Profiles of Optical Turbulence (Cn2) from Reanalysis Using Deep Learning* (Version 0.2.1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23733024
