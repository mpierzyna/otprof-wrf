# OTProf code: dataset pipeline, SQF-H model, paper figures

Everything needed to build the dataset from the WRF runs in [`../data`](../data), train and run the height-aware
Squeezeformer (SQF-H), and reproduce the figures and score tables of the paper.

The processed datasets, the trained models (checkpoints and fitted scalers) and the predictions are on
[Zenodo](https://zenodo.org/records/20733692).

## Layout

```
code/
├── dataset/                 # DVC pipeline: WRF post-processing output + ERA5 -> datasets
│   ├── dvc.yaml, params.yaml
│   ├── step_0_link_wrf.py           link ../data/<domain>/wrf/sim_*/4_postproc/wrfout*_cn2.nc
│   ├── step_1_download_era5.py      ERA5 pressure levels + surface from the ARCO ERA5 store
│   ├── step_2a_align_era5_wrf.py    spatial / temporal alignment of ERA5 and WRF
│   ├── step_3a_make_features.py     gradients, wind, W71 surface CT2, astroclimate parameters
│   ├── step_3b_filter_cn2.py        discard invalid / extreme Cn2 profiles
│   ├── step_4c_interpolate_wrf_era5pl.py  WRF -> ERA5 pressure levels (WRF PL inputs)
│   └── data/<domain>/       # outputs: 1_raw/ 2_aligned/ 3_features/ 4_features_interp/
├── src/otprof/              # package
│   ├── pipelines/           # data pipelines: loading, height features, splits, scalers
│   ├── model.py             # SQF-H model
│   ├── lit.py               # LightningModule (loss, optimizer)
│   ├── experiments.py       # the three experiments of the paper
│   ├── predict.py           # inference, ERA5 input swap, HV+W71 reference, output quantile mapping
│   └── base.py, datasets.py, transform.py, losses.py, logging.py, types.py
├── src/research_tools/      # vendored subset: optical-turbulence integrals, HV profile, YAML config, HTML report
├── scripts/
│   ├── train.py             # train wrf_pl, wrf_native, era5_pl_direct on nl
│   ├── predict_nl.py        # nl 2017 test predictions + nl output QM
│   ├── predict_canaries.py  # zero-shot Canary Islands predictions + output QM
│   └── predict_era5.py      # predictions from ERA5 for any time and location
├── paper/
│   ├── analysis.py          # loading of the predictions, metrics, structure functions
│   └── figures.py           # all figures and score tables -> paper/plots/
└── runs/                    # (not in git) models/<experiment>/, preds/{nl, canaries/<group>}/
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). From `code/`:

```bash
uv sync
```

This installs the `otprof` package (editable) with the pinned model stack and DVC. All commands below run
from `code/`. A GPU is used for training and prediction if available.

The WRF campaigns in [`../data`](../data) use their own environment (the repository root project with
[`wrf-massive`](https://github.com/mpierzyna/wrf-massive)), see the READMEs there.

## 1. Dataset

Domains: `nl` (Netherlands, 2017-2020, training) and `canaries` (Canary Islands, one 5-day run per month of
2017, out of domain). After running the WRF campaign of a domain (`../data/<domain>/wrf`, post-processing
stage included):

```bash
dvc repro                                # all stages, all domains
dvc repro 'make_features@canaries'       # one stage of one domain (and what it depends on)
```

ERA5 is read anonymously from the [ARCO ERA5](https://github.com/google-research/arco-era5) store on Google
Cloud. The model reads per domain `3_features/{wrf_filtered,era5_pl,era5_sfc}.nc` and
`4_features_interp/wrf_filtered_era5pl.nc`. To skip the pipeline, place the files from Zenodo there.

## 2. Models

| experiment       | input -> output                                    | run dir                    |
|------------------|----------------------------------------------------|----------------------------|
| `wrf_pl`         | WRF on ERA5 pressure levels -> native WRF (decoupled training; ERA5 PL at inference) | `runs/models/wrf_pl` |
| `wrf_native`     | native WRF -> native WRF (control)                 | `runs/models/wrf_native`   |
| `era5_pl_direct` | ERA5 PL -> native WRF (no decoupling, ablation)    | `runs/models/era5_pl_direct` |

Train on nl (2019-2020 train, 2018 validation, 2017 test):

```bash
uv run python scripts/train.py
```

or put the trained models from Zenodo into `runs/models/`. Each run dir holds `checkpoints/` (the five
best by validation loss; prediction uses the best), `data_pipeline/` (config and fitted scalers),
`exp_config.yml` and `hparams.yaml`. `wrf_pl` also holds the fitted nl output quantile mapping
(`oqm_lcn2.joblib`, written by `scripts/predict_nl.py`).

## 3. Predictions

```bash
uv run python scripts/predict_nl.py         # -> runs/preds/nl/
uv run python scripts/predict_canaries.py   # -> runs/preds/canaries/{A,B,C}/ (needs the nl predictions)
```

`predict_nl.py` predicts the nl test year with every model, the decoupled model fed with ERA5 PL, the
Hufnagel-Valley reference (HV+W71), and fits the bulk lcn2 output quantile mapping (nl-oQM) on the
ERA5-driven predictions of the training years. `predict_canaries.py` runs the nl model zero-shot on the
three Canary Islands month groups and applies the nl-oQM and the canaries oQM (fit on one group, applied to
the two others). Existing predictions are skipped.

## 4. Figures

```bash
uv run python paper/figures.py              # -> paper/plots/
```

`NL_SUBSAMPLE` in `paper/figures.py` subsamples the nl test set for quick previews (1 for the paper).

## 5. Predictions from ERA5 for your own times and locations

```bash
uv run python scripts/predict_era5.py       # -> runs/preds/era5/sqfh_era5.nc
```

Set the locations, times, output heights and whether to apply the nl-oQM at the top of the script. It reads ERA5
for the nearest grid points directly from the ARCO store and needs only `runs/models/wrf_pl/` (no dataset). The output
holds the lcn2, QKE, TSQ and EL_PBL profiles (time, location, height), the nl-oQM corrected lcn2, and r0 and lsi2.
Every requested time step downloads the global ERA5 fields (~0.7 GB), so keep the time range short. Two differences
to the paper setup: the terrain height input is ERA5's (training used the WRF terrain), and outside the nl training
domain (lat 50.25-54.0, lon 2.75-8.25) the model runs zero-shot, for which the script warns.
