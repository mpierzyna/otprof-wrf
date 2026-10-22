"""SQF-H Cn2 profiles from ERA5 for times and locations of your choice.

Reads ERA5 pressure-level and surface data at the grid points nearest to LOCATIONS from the ARCO ERA5 store (Google
Cloud, anonymous access), builds the model inputs as the dataset pipeline does for the training data (dataset/step_2a,
step_3a and otprof.pipelines), and predicts with the best checkpoint of the decoupled model (wrf_pl) at the heights
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
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from research_tools.ot import most, ot

from otprof import RUNS_ROOT
from otprof.datasets import ProfileDataset
from otprof.lit import LitSqueezeFormerH
from otprof.pipelines.shared import P_LEVELS_ERA5, WRF_TF_CONFIG, Z_AGL_WRF, _lz, add_X_time_features
from otprof.predict import get_best_ckpt

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
BATCH_SIZE = 4096

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_VARS = [
    # Pressure levels
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    # Surface
    "geopotential_at_surface",
    "surface_pressure",
    "land_sea_mask",
    "boundary_layer_height",
    "friction_velocity",
    "instantaneous_surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
# ERA5 grid points of the nl training data (lat, lon range in degrees)
NL_TRAIN_BOX = {"lat": (50.25, 54.0), "lon": (2.75, 8.25)}
# The W71 surface CT2 clips the Obukhov length to its 2.5 / 97.5 percentiles over the dataset (dataset/step_3a); these
# are the bounds of the nl ERA5 data, so lct2_w71f is computed as for the training inputs.
W71_L_BOUNDS_NL = (-1003.6894, 1003.8965)  # m


def load_era5(locations: dict[str, tuple[float, float]], times: pd.DatetimeIndex) -> xr.Dataset:
    """ERA5 at the grid points nearest to `locations` (dim loc), on the 31 pressure levels 1000..20 hPa, bottom to
    top (dim level)."""
    ds = xr.open_zarr(ARCO_ERA5, chunks={}, storage_options=dict(token="anon"))
    valid_until = ds.attrs.get("valid_time_stop_era5t")
    ds = ds[ERA5_VARS].sel(time=times)
    print(f"Downloading {len(times)} global ERA5 time steps (~{ds.nbytes / 1e9:.1f} GB uncompressed)...")

    lat = xr.DataArray([lat for lat, _ in locations.values()], dims="loc")
    lon = xr.DataArray([lon % 360 for _, lon in locations.values()], dims="loc")  # ARCO uses 0..360
    levels = [int(p) for p in P_LEVELS_ERA5 / 100] + [20]  # 20 hPa only for the gradient of the top half level
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest").sel(level=levels).load()

    if ds["temperature"].isnull().any():
        raise ValueError(f"No ERA5 data for some of the requested times (ARCO store valid until {valid_until}).")
    return ds.assign_coords(
        loc=list(locations),
        lat=ds["latitude"],
        lon=((ds["longitude"] + 180) % 360) - 180,
    ).drop_vars(["latitude", "longitude"])


def make_features(ds: xr.Dataset) -> xr.Dataset:
    """SQF-H inputs (time, loc, bottom_top) on the 30 ERA5 pressure levels 1000..30 hPa. Gradients live on the half
    level above each pressure level."""
    p = (ds["level"] * 100).drop_vars("level")  # Pa
    ds = ds.drop_vars("level")  # positional levels, bottom to top
    full = {"level": slice(0, len(P_LEVELS_ERA5))}  # 1000..30 hPa; 20 hPa only enters the top gradient
    p_sfc = ds["surface_pressure"]
    z = ds["geopotential"] / 9.81
    z_sfc = ds["geopotential_at_surface"].mean("time") / 9.81
    z_agl = (z - z_sfc).isel(full)
    th = (ds["temperature"] * (100000 / p) ** ((1.4 - 1) / 1.4)).astype(np.float32)

    # Vertical gradients between consecutive levels
    dz = z.diff("level")
    du_dz = ds["u_component_of_wind"].diff("level") / dz
    dv_dz = ds["v_component_of_wind"].diff("level") / dz
    S = np.sqrt(du_dz**2 + dv_dz**2).astype(np.float32)
    dth_dz = (th.diff("level") / dz).astype(np.float32)

    # Surface fluxes in WRF convention (upward positive, W/m2) and the W71 surface CT2
    ust = ds["friction_velocity"]
    hfx = -ds["instantaneous_surface_sensible_heat_flux"]
    tk2 = ds["2m_temperature"]
    shfx = hfx / 1.216e3  # W/m2 -> K m/s
    L = -(ust**3) * tk2 / (0.4 * shfx * 9.81)
    ct2_w71f = most.w71_flux_ct2(T_st=-shfx / ust, z=2.0, L=L, L_bounds=W71_L_BOUNDS_NL).clip(1e-8, None)

    ds_X = xr.Dataset(
        {
            "z_agl": z_agl,
            "p": p.isel(full),
            "u": ds["u_component_of_wind"].isel(full),
            "v": ds["v_component_of_wind"].isel(full),
            "th": th.isel(full),
            "S": S,
            "dth_dz": dth_dz,
            "ust": ust,
            "hfx": hfx,
            "lh": -ds["surface_latent_heat_flux"] / 3600,  # J/m2 per hour -> W/m2
            "blh": ds["boundary_layer_height"],
            "u10": ds["10m_u_component_of_wind"],
            "v10": ds["10m_v_component_of_wind"],
            "tk2": tk2,
            "slp": ds["mean_sea_level_pressure"] / 100,  # hPa
            "lsm": ds["land_sea_mask"],
            "lct2_w71f": np.log10(ct2_w71f),
            "valid": (p.isel(full) < p_sfc).astype("float32"),
            "lz": _lz(z_agl),
            "z_sfc": z_sfc,
            "lp_sfc": np.log(p_sfc),
        }
    ).rename(level="bottom_top")
    X_time, _ = add_X_time_features(ds_X, None)
    return ds_X.assign(X_time)


def predict_sqfh(ds_X: xr.Dataset, z_agl: xr.DataArray, model_dir: pathlib.Path) -> xr.Dataset:
    """Predict the targets of the model in `model_dir` (best checkpoint) from the inputs `ds_X` (time, loc,
    bottom_top) at the heights `z_agl` (dim level, optionally also time and loc). Returns (time, loc, level)."""
    exp_config = yaml.safe_load((model_dir / "exp_config.yml").read_text())
    scaler_X = joblib.load(model_dir / "data_pipeline" / "scaler_X.joblib")
    scaler_y = joblib.load(model_dir / "data_pipeline" / "scaler_y.joblib")
    lit = LitSqueezeFormerH.load_from_checkpoint(get_best_ckpt(model_dir), map_location="cpu")
    model = lit.model._orig_mod.to(DEVICE).eval()  # uncompiled module, same weights

    # Samples as in the training data: stacked time and location
    times, locs = ds_X["time"], ds_X["loc"]
    z_agl = z_agl.broadcast_like(ds_X["lsm"]).transpose("time", "loc", "level")
    ds_X = ds_X.stack(sample=("time", "loc")).transpose("sample", ...)
    ds_frc = xr.Dataset({"z_agl": z_agl, "lz": _lz(z_agl)}).stack(sample=("time", "loc")).transpose("sample", ...)
    data = ProfileDataset(
        ds_X=scaler_X.transform(ds_X[exp_config["features"]]),
        vars_X=exp_config["features"],
        ds_y=scaler_y.transform(ds_frc),  # lz is not scaled
        vars_y=[],
        vars_y_frc=exp_config["forcings"],
    )
    assert data.vars_X.index("valid") == lit.hparams.idx_valid and data.vars_X.index("lz") == lit.hparams.idx_lz

    with torch.no_grad():
        y = [
            model(data.X[i : i + BATCH_SIZE].to(DEVICE), data.y_frc[i : i + BATCH_SIZE, :, 1].to(DEVICE))[0].cpu()
            for i in range(0, len(data), BATCH_SIZE)
        ]
    y = torch.concat(y).numpy().reshape(times.size, locs.size, z_agl.sizes["level"], -1)

    # Unscale and invert the log transforms of the targets
    ds_y = xr.Dataset({v: (("time", "loc", "level"), y[..., i]) for i, v in enumerate(exp_config["targets"])})
    ds_y = scaler_y.inverse_transform(ds_y)
    for v, (_, fn_inv) in WRF_TF_CONFIG.items():
        if v in ds_y:
            ds_y[v] = fn_inv(ds_y[v])
    return ds_y.assign_coords(time=times, loc=locs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for name, (lat, lon) in LOCATIONS.items():
        lat_g, lon_g = round(lat * 4) / 4, round((((lon + 180) % 360) - 180) * 4) / 4  # nearest ERA5 grid point
        (lat_min, lat_max), (lon_min, lon_max) = NL_TRAIN_BOX["lat"], NL_TRAIN_BOX["lon"]
        if not (lat_min <= lat_g <= lat_max and lon_min <= lon_g <= lon_max):
            warnings.warn(
                f"Location '{name}' ({lat}, {lon}) is outside the nl training domain (lat {lat_min}..{lat_max}, "
                f"lon {lon_min}..{lon_max}). The model runs zero-shot there and the nl-oQM may not apply; "
                "expect larger errors, in particular over complex terrain.",
                stacklevel=1,
            )

    ds_era5 = load_era5(LOCATIONS, TIMES)
    ds_X = make_features(ds_era5)
    z_agl = xr.DataArray(np.asarray(Z_AGL, dtype=np.float32), dims="level")
    ds_pred = predict_sqfh(ds_X, z_agl, MODEL_DIR)

    # nl output quantile mapping of lcn2 (one transfer function for all levels)
    if APPLY_OQM:
        qm = joblib.load(MODEL_DIR / "oqm_lcn2.joblib")[0]
        ds_pred["lcn2_oqm"] = qm.transform(ds_pred[["lcn2"]])["lcn2"].astype(np.float32)

    # Integrated parameters (1550 nm, plane wave) from the profiles, as in the paper
    for v in [v for v in ["lcn2", "lcn2_oqm"] if v in ds_pred]:
        cn2, suffix = 10 ** ds_pred[v], v.removeprefix("lcn2")
        ds_pred[f"r0{suffix}"] = ot.fried_r0_xr(cn2=cn2, z=z_agl, dim="level") * 100
        ds_pred[f"lsi2{suffix}"] = np.log10(ot.scint_index_xr(cn2=cn2, z=z_agl, dim="level", mode="plane"))

    units = {"lcn2": "log10(m^-2/3)", "QKE": "m2 s-2", "TSQ": "K2", "EL_PBL": "m", "r0": "cm", "lsi2": "1"}
    for v in ds_pred:
        ds_pred[v].attrs["units"] = units[v.removesuffix("_oqm")]
    ds_pred = ds_pred.assign_coords(
        z_agl=z_agl.assign_attrs(units="m", long_name="Height above ERA5 ground"),
        lat=ds_era5["lat"],
        lon=ds_era5["lon"],
        z_sfc=ds_X["z_sfc"].assign_attrs(units="m", long_name="ERA5 terrain height"),
        lsm=ds_X["lsm"].mean("time").assign_attrs(long_name="ERA5 land-sea mask"),
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_pred.to_netcdf(OUT_FILE)
    print(ds_pred)
    print(f"Predictions written to {OUT_FILE}")
