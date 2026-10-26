# OTProf-WRF

Simulate horizontal and vertical profiles of atmospheric (optical) turbulence with the
[Weather Research and Forecasting (WRF) model](https://github.com/wrf-model/wrf). The simulations are orchestrated by [`wrf-massive`](https://github.com/mpierzyna/wrf-massive),
which allows to parameterize WRF simulations and their entire pipeline (forcing download, WPS, 
WRF, Cn2 postprocessing).

The domains [`nl`](./domains/nl/) and [`canaries`](./domains/canaries/) are used in

> Pierzyna, Maximilian, et al. “OTProf: Estimating High-Resolution Profiles of Optical Turbulence ($C_n^2$) from Reanalysis Using Deep Learning.” arXiv:2604.09346, arXiv, 10 Apr. 2026. arXiv.org, https://doi.org/10.48550/arXiv.2604.09346.

The full model code accompanying the OTProf publication can be found [here](https://github.com/mpierzyna/otprof).


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
