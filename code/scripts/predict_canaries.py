"""Zero-shot predictions of the nl-trained decoupled model on the Canary Islands, and their output quantile
mapping (oQM). Needs the nl predictions (scripts/predict_nl.py) for the nl-oQM.

The canaries WRF run covers one 5-day period per month of 2017. Each month is an independent run, and the
months are split into three groups of 20 days:
  A = months 1, 4, 7, 10,   B = months 2, 5, 8, 11,   C = months 3, 6, 9, 12
Weights and scalers come from nl, so all 60 days are out of sample for the model.

Writes runs/preds/canaries/<group>/:
  wrfpl_ref.nc, wrfpl_*      WRF interpolated to ERA5 PL (training modality, truth = native WRF), HV+W71
  era5pl_*                   real ERA5 PL (the deployment path)
  era5pl_oqmnl_y_pred.nc     era5pl + nl-oQM (fit on nl, no canaries data, see predict_nl.py)
  era5pl_oqmfit<X>_y_pred.nc era5pl + canaries oQM fit on group X (one map for WRF land and one for WRF
                             sea locations), applied to the two other groups only (out of sample)

Existing predictions are skipped. Run from code/:  uv run python scripts/predict_canaries.py
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from otprof import RUNS_ROOT
from otprof.experiments import get_exp
from otprof.predict import output_qm_lcn2, predict, predict_ref, wrf_to_era5

DOMAIN = "canaries"
MODEL = RUNS_ROOT / "models" / "wrf_pl"
NL_DIR = RUNS_ROOT / "preds" / "nl"
OUT_ROOT = RUNS_ROOT / "preds" / DOMAIN
GROUP_MONTHS = {"A": [1, 4, 7, 10], "B": [2, 5, 8, 11], "C": [3, 6, 9, 12]}


def zeroshot(group: str) -> None:
    """WRF PL and ERA5 PL predictions and the HV+W71 reference of the months of `group`."""
    out_dir = OUT_ROOT / group
    months = GROUP_MONTHS[group]
    exp = get_exp("wrf_pl", domain=DOMAIN, scaler_dir=MODEL / "data_pipeline")
    # nl scalers, canaries months: only the test stage is used
    exp.dp.sel_test = lambda p: {"sample": p.ds_X["time"].dt.month.isin(months)}
    predict_ref(exp, out_dir=out_dir, out_basename="wrfpl")
    predict(exp, MODEL, out_dir=out_dir, out_basename="wrfpl")
    exp_era5 = wrf_to_era5(exp, scaler_dir=MODEL / "data_pipeline")
    predict(exp_era5, MODEL, out_dir=out_dir, out_basename="era5pl")


def output_qm(fit: str) -> None:
    """Fit the bulk lcn2 oQM on `fit` ("nl" or a canaries group) and apply it to the ERA5 predictions of the
    canaries groups it was not fit on."""
    # WRF land-sea mask per location (ERA5's own mask marks only half of the WRF land points as land)
    X_wrf = xr.open_dataset(OUT_ROOT / "C" / "wrfpl_X_test.nc")
    land_locs = np.unique(X_wrf["loc"].values[X_wrf["lsm"].values > 0.5])

    if fit == "nl":
        truth_fit = xr.open_dataset(NL_DIR / "era5pl_trainyears_y_test.nc")[["lcn2"]].load()
        pred_fit = xr.open_dataset(NL_DIR / "era5pl_trainyears_y_pred.nc")[["lcn2"]].load()
        apply_to, tag = list(GROUP_MONTHS), "nl"
    else:
        truth_fit = xr.open_dataset(OUT_ROOT / fit / "era5pl_y_test.nc")[["lcn2"]].load()
        pred_fit = xr.open_dataset(OUT_ROOT / fit / "era5pl_y_pred.nc")[["lcn2"]].load()
        apply_to, tag = [g for g in GROUP_MONTHS if g != fit], f"fit{fit}"

    for g in apply_to:
        pred = xr.open_dataset(OUT_ROOT / g / "era5pl_y_pred.nc").load()
        if fit == "nl":  # one map
            groups_fit = groups_test = None
        else:  # land and sea maps
            groups_fit = np.isin(truth_fit["loc"].values, land_locs)
            groups_test = np.isin(pred["loc"].values, land_locs)
        out = pred.copy()
        lcn2 = output_qm_lcn2(truth_fit, pred_fit, pred, groups_fit=groups_fit, groups_test=groups_test)
        out["lcn2"] = (("sample", "level"), lcn2)
        out.to_netcdf(OUT_ROOT / g / f"era5pl_oqm{tag}_y_pred.nc")
        print(f"Output QM (fit {fit}) applied to group {g}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for group in GROUP_MONTHS:
        if not (OUT_ROOT / group / "era5pl_y_pred.nc").exists():
            zeroshot(group)
    for fit in ["nl", *GROUP_MONTHS]:
        output_qm(fit)
    print(f"All canaries predictions written to {OUT_ROOT}")
