"""SQF-H Cn2 profiles from ERA5 for times and locations of your choice.

Reads ERA5 pressure-level and surface data at the grid points nearest to LOCATIONS from the ARCO ERA5 store (Google
Cloud, anonymous access), builds the model inputs as the dataset pipeline does for the training data (dataset/step_2a,
step_3a and otprof.pipelines; see otprof.predict.make_pl_inputs), and predicts with the best checkpoint of the decoupled model (wrf_pl) at the heights
Z_AGL above the ERA5 ground. Optionally applies the nl output quantile mapping (nl-oQM) to lcn2 and computes r0 and
lsi2 from the profiles. Needs runs/models/wrf_pl/ (trained model with oqm_lcn2.joblib, see scripts/predict_nl.py, or
from Zenodo).

The ARCO store is chunked by time step: every time step downloads the global fields (~0.7 GB uncompressed for all
variables), independent of the number of locations. Keep TIMES short.

Differences to the paper setup:
- The terrain height input z_sfc is ERA5's instead of the WRF terrain of the training data (no WRF here).
- The model was trained on nl (2019-2020); outside the nl training box it runs zero-shot, and the nl-oQM was fitted on
  nl only. A warning is issued for such locations.

Edit the settings below, then run from code/:  uv run python scripts/predict_era5.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import xarray as xr

from otprof import RUNS_ROOT
from otprof.pipelines.shared import P_LEVELS_ERA5, Z_AGL_WRF
from otprof.predict import make_pl_inputs, predict_profiles, warn_outside_nl

## Settings
LOCATIONS = {  # name: (lat, lon) in degrees north / east; the nearest ERA5 grid point (0.25 deg) is used
    "cabauw": (51.97, 4.93),
    "north_sea": (53.25, 4.25),
}
TIMES = pd.date_range("2024-07-01 00:00", "2024-07-01 18:00", freq="6h")  # UTC, ERA5 is hourly
Z_AGL = Z_AGL_WRF  # output heights [m above ground], increasing; default: mean heights of the 99 WRF training levels
APPLY_OQM = True  # add lcn2_oqm (nl-oQM) and the r0, lsi2 computed from it
MODEL_DIR = RUNS_ROOT / "models" / "wrf_pl"
OUT_FILE = RUNS_ROOT / "preds" / "era5" / "sqfh_era5.nc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_RENAME = {
    # Pressure levels
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "tk",
    # Surface
    "geopotential_at_surface": "z_sfc",
    "surface_pressure": "p_sfc",
    "land_sea_mask": "lsm",
    "boundary_layer_height": "blh",
    "friction_velocity": "ust",
    "instantaneous_surface_sensible_heat_flux": "hfx",
    "surface_latent_heat_flux": "lh",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "tk2",
    "mean_sea_level_pressure": "slp",
}


def load_era5(locations: dict[str, tuple[float, float]], times: pd.DatetimeIndex) -> xr.Dataset:
    """ERA5 at the grid points nearest to `locations` (dim loc), on the 31 pressure levels 1000..20 hPa, bottom to
    top (dim level), in the conventions of otprof.predict.make_pl_inputs (as dataset/step_2a and step_3a)."""
    ds = xr.open_zarr(ARCO_ERA5, chunks={}, storage_options=dict(token="anon"))
    valid_until = ds.attrs.get("valid_time_stop_era5t")
    ds = ds[list(ERA5_RENAME)].sel(time=times)
    print(f"Downloading {len(times)} global ERA5 time steps (~{ds.nbytes / 1e9:.1f} GB uncompressed)...")

    lat = xr.DataArray([lat for lat, _ in locations.values()], dims="loc")
    lon = xr.DataArray([lon % 360 for _, lon in locations.values()], dims="loc")  # ARCO uses 0..360
    levels = [int(p) for p in P_LEVELS_ERA5 / 100] + [20]  # 20 hPa only for the gradient of the top half level
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest").sel(level=levels).load()
    if ds["temperature"].isnull().any():
        raise ValueError(f"No ERA5 data for some of the requested times (ARCO store valid until {valid_until}).")

    ds = ds.rename(ERA5_RENAME)
    ds["z"] = ds["z"] / 9.81
    ds["z_sfc"] = ds["z_sfc"].mean("time") / 9.81  # static
    ds["hfx"] = -ds["hfx"]  # ERA5 fluxes are downward positive
    ds["lh"] = -ds["lh"] / 3600  # J/m2 per hour -> W/m2
    ds["slp"] = ds["slp"] / 100  # hPa
    return ds.assign_coords(
        loc=list(locations),
        lat=ds["latitude"],
        lon=((ds["longitude"] + 180) % 360) - 180,
    ).drop_vars(["latitude", "longitude"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warn_outside_nl(LOCATIONS)

    ds_era5 = load_era5(LOCATIONS, TIMES)
    ds_X = make_pl_inputs(ds_era5)
    z_agl = xr.DataArray(np.asarray(Z_AGL, dtype=np.float32), dims="level")
    ds_pred = predict_profiles(ds_X, z_agl, MODEL_DIR, apply_oqm=APPLY_OQM, device=DEVICE)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_pred.to_netcdf(OUT_FILE)
    print(ds_pred)
    print(f"Predictions written to {OUT_FILE}")
