"""Predictions of the nl 2017 test year (stepwise decoupling) and the nl output quantile mapping (oQM).

Writes runs/preds/nl/<case>_{X_test,y_test,y_pred}.nc (inputs, truth = native WRF, predictions):
  wrfnative       WRF native -> WRF native, regression only (control)
  wrfpl           WRF PL -> WRF native (decoupled model, training modality), plus wrfpl_ref.nc (HV+W71)
  era5pl          decoupled model fed with ERA5 PL
  era5pl_direct   model trained on ERA5 PL directly (no decoupling)
  era5pl_oqm      era5pl + bulk lcn2 oQM, fit on the ERA5-driven predictions of the training years
                  (era5pl_trainyears, 2019-2020) against native WRF. Also the nl-oQM of the canaries.
                  The fitted mapping is saved to runs/models/wrf_pl/oqm_lcn2.joblib (used by predict_era5.py).

Existing outputs are skipped. Run from code/:  uv run python scripts/predict_nl.py
"""

from __future__ import annotations

import logging

import xarray as xr

from otprof import RUNS_ROOT
from otprof.experiments import get_exp
from otprof.predict import output_qm_lcn2, predict, predict_ref, wrf_to_era5

MODELS = RUNS_ROOT / "models"
OUT_DIR = RUNS_ROOT / "preds" / "nl"


def done(basename: str) -> bool:
    return (OUT_DIR / f"{basename}_y_pred.nc").exists()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not done("wrfnative"):
        predict(get_exp("wrf_native"), MODELS / "wrf_native", out_dir=OUT_DIR, out_basename="wrfnative")

    wrf_pl = None
    if not done("wrfpl"):
        wrf_pl = get_exp("wrf_pl")
        predict(wrf_pl, MODELS / "wrf_pl", out_dir=OUT_DIR, out_basename="wrfpl")
        predict_ref(wrf_pl, out_dir=OUT_DIR, out_basename="wrfpl")

    if not done("era5pl"):
        wrf_pl = wrf_pl or get_exp("wrf_pl")
        predict(wrf_to_era5(wrf_pl), MODELS / "wrf_pl", out_dir=OUT_DIR, out_basename="era5pl")

    if not done("era5pl_direct"):
        predict(get_exp("era5_pl_direct"), MODELS / "era5_pl_direct", out_dir=OUT_DIR, out_basename="era5pl_direct")

    # oQM fit data: ERA5-driven predictions of the training years
    if not done("era5pl_trainyears"):
        wrf_pl = wrf_pl or get_exp("wrf_pl")
        exp_era5_train = wrf_to_era5(wrf_pl)
        exp_era5_train.dp.sel_test = exp_era5_train.dp.sel_train  # predict() predicts the test stage
        predict(exp_era5_train, MODELS / "wrf_pl", out_dir=OUT_DIR, out_basename="era5pl_trainyears")

    truth_fit = xr.open_dataset(OUT_DIR / "era5pl_trainyears_y_test.nc")[["lcn2"]].load()
    pred_fit = xr.open_dataset(OUT_DIR / "era5pl_trainyears_y_pred.nc").load()
    pred_test = xr.open_dataset(OUT_DIR / "era5pl_y_pred.nc").load()
    out = pred_test.copy()
    oqm_path = MODELS / "wrf_pl" / "oqm_lcn2.joblib"
    out["lcn2"] = (("sample", "level"), output_qm_lcn2(truth_fit, pred_fit, pred_test, save_path=oqm_path))
    out.to_netcdf(OUT_DIR / "era5pl_oqm_y_pred.nc")
    print(f"All nl predictions written to {OUT_DIR}")
