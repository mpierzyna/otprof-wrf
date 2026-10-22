from typing import Tuple, Dict

import numpy as np
import xarray as xr


def add_X_time_features(ds_X: xr.Dataset, _) -> Tuple[Dict[str, xr.DataArray], Dict]:
    """Add time-based features to input dataset."""
    t = ds_X["time"].dt
    X_fe = {
        "hr_sin": np.sin(2 * np.pi * t.hour / 24),
        "hr_cos": np.cos(2 * np.pi * t.hour / 24),
        "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365),
    }
    return X_fe, {}


# Engineered features
FE_SCALER_CONFIG = {
    "hr_sin": (None, "minmax", None),
    "hr_cos": (None, "minmax", None),
    "doy_sin": (None, "minmax", None),
    "doy_cos": (None, "minmax", None),
}

# This is used for all ERA5 pipelines (PL and ML)
ERA5_SCALER_CONFIG = {
    # both PL and ML
    "z": (None, "minmax", None),
    "z_rel": (None, "quantile", (0, 0.99)),
    "z_agl": (None, "quantile", (0, 0.99)),
    "p": (None, "minmax", None),
    "u": (None, "quantile", (0.01, 0.99)),
    "v": (None, "quantile", (0.01, 0.99)),
    "m": (None, "minmax", None),
    "d_sin": (None, "minmax", None),
    "d_cos": (None, "minmax", None),
    "w": (None, "quantile", (0.01, 0.99)),
    "th": (None, "minmax", None),
    "dth_dz": (None, "quantile", (0.01, 0.99)),
    "S": (None, "quantile", (0, 0.99)),
    # sfc
    "ust": (None, "quantile", (0, 0.99)),
    "hfx": (None, "quantile", (0.01, 0.99)),
    "lh": (None, "quantile", (0.01, 0.99)),
    "blh": (None, "quantile", (0.00, 0.99)),
    "u10": (None, "quantile", (0.01, 0.99)),
    "v10": (None, "quantile", (0.01, 0.99)),
    "tk2": (None, "quantile", (0.01, 0.99)),
    "lsm": (None, "minmax", None),
    "boundary_layer_dissipation": (None, "quantile", (0.0, 0.99)),
    "angle_of_sub_gridscale_orography": (None, "minmax", None),
    "anisotropy_of_sub_gridscale_orography": (None, "minmax", None),
    "slp": (None, "minmax", None),  # sea level pressure
    "lct2_w71f": (None, "quantile", (0.00, 0.995)),  # already clipped on left
}
ERA5_SCALER_CONFIG_PL = {
    # PL only
    "pv": (None, "quantile", (0.01, 0.99)),
}
ERA5_VARS_PL = list(ERA5_SCALER_CONFIG.keys()) + list(ERA5_SCALER_CONFIG_PL.keys())

# Combine all configs here
ERA5_SCALER_CONFIG = {
    **ERA5_SCALER_CONFIG,
    **ERA5_SCALER_CONFIG_PL,
    **FE_SCALER_CONFIG,
}

WRF_SCALER_CONFIG = {
    # Turbulent profiles
    "lcn2": (None, "quantile", (0.005, 0.995)),
    "QKE": (None, "minmax", None),  # already clipped on lower end
    "TSQ": (None, "quantile", (0.005, 0.995)),
    "EL_PBL": (None, "quantile", (0.005, 0.995)),
    # Mean profiles
    "z": (None, "minmax", None),
    "z_agl": (None, "minmax", None),
    "p": (None, "minmax", None),
    "m": (None, "quantile", (0.0, 0.99)),
    "d_sin": (None, "minmax", None),
    "d_cos": (None, "minmax", None),
    "u": (None, "quantile", (0.01, 0.99)),
    "v": (None, "quantile", (0.01, 0.99)),
    "w": (None, "quantile", (0.01, 0.99)),
    "th": (None, "minmax", None),
    # Gradients
    "dth_dz": (None, "quantile", (0.01, 0.99)),
    "S": (None, "quantile", (0, 0.99)),
    # Surface
    "ust": (None, "quantile", (0, 0.99)),
    "hfx": (None, "quantile", (0.01, 0.99)),
    "lh": (None, "quantile", (0.01, 0.99)),
    "blh": (None, "quantile", (0.00, 0.99)),
    "u10": (None, "quantile", (0.01, 0.99)),
    "v10": (None, "quantile", (0.01, 0.99)),
    "tk2": (None, "quantile", (0.01, 0.99)),
    "slp": (None, "minmax", None),
    "lsm": (None, "minmax", None),
    # Engineered features
    "lct2_w71f": (None, "quantile", (0.00, 0.995)),  # already clipped on left
    **FE_SCALER_CONFIG,
}

# Have lambda in individual lines for serialization to work.
WRF_TF_CONFIG = {
    "QKE": (
        lambda qke: np.log10(qke.clip(min=1e-8)),
        lambda x: 10**x,
    ),
    "TSQ": (
        lambda tsq: np.log10(tsq.clip(min=1e-8)),
        lambda x: 10**x,
    ),
    "EL_PBL": (
        lambda el_pbl: np.log10(el_pbl.clip(min=1e-1)),
        lambda x: 10**x,
    ),
}


## Height-aware features (SQF-H)

# Nominal ERA5 pressure levels [Pa] of the 30-level PL input grid (1000..30 hPa), bottom to top.
# Same levels as `4_features_interp/wrf_filtered_era5pl.nc` (dataset/step_4c) and the WRF-matched ERA5 PL data.
P_LEVELS_ERA5 = 100.0 * np.array(
    [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550,
     500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30],
)  # fmt: skip

# Mean height above ground [m] of the 99 native WRF output levels (targets) over the nl training years
# 2019-2020. Default query heights for predictions without a WRF target grid (scripts/predict_era5.py).
Z_AGL_WRF = np.array(
    [8.3, 25.8, 45.1, 65.2, 86.6, 110.2, 135.1, 161.8, 190.3, 220.5, 253.4, 288.6, 326.0, 366.2, 409.2, 455.0,
     503.6, 555.5, 611.2, 670.4, 733.5, 800.6, 872.3, 948.6, 1029.2, 1115.1, 1206.5, 1303.4, 1406.2, 1514.9,
     1630.2, 1751.9, 1880.3, 2016.0, 2159.4, 2310.2, 2468.3, 2635.2, 2810.5, 2993.7, 3185.0, 3385.7, 3595.4,
     3813.7, 4040.8, 4276.7, 4521.3, 4774.3, 5036.5, 5307.5, 5586.6, 5874.2, 6169.2, 6471.3, 6781.0, 7097.9,
     7421.4, 7751.1, 8087.4, 8428.6, 8774.2, 9125.3, 9482.6, 9844.1, 10209.3, 10581.1, 10959.3, 11343.0,
     11732.8, 12131.0, 12538.1, 12952.4, 13375.8, 13807.8, 14247.9, 14695.2, 15148.8, 15613.7, 16086.1,
     16565.9, 17055.1, 17553.6, 18064.4, 18582.8, 19112.5, 19657.7, 20218.7, 20795.3, 21386.9, 22000.8,
     22637.1, 23287.0, 23970.3, 24687.8, 25445.5, 26260.8, 27108.6, 28033.6, 29069.3],
)  # fmt: skip

# Warped height lz = log1p(z_agl / LZ_REF) / LZ_NORM, ~[0, 1] over 0..30 km. lz is used as the
# common height coordinate of inputs (X) and outputs (y) for the model's physical level
# interpolation, so it must NOT be scaled per dataset: X and y scalers are fit separately and
# would put the two sides into different affine spaces. Hence fixed constants here and no
# scaler entry for lz (DatasetScaler passes unconfigured variables through).
LZ_REF = 10.0  # m
LZ_NORM = float(np.log1p(30_000.0 / LZ_REF))


def _lz(z_agl: xr.DataArray) -> xr.DataArray:
    # Clip: ERA5 extrapolates below its orography (z_agl < 0); those levels are masked anyway.
    return np.log1p(z_agl.clip(min=0) / LZ_REF) / LZ_NORM


def _z_sfc_wrf(ds_y: xr.Dataset) -> xr.DataArray:
    # True terrain height of the target column (also used for ERA5 inputs, see add_height_features_era5pl).
    # Native-WRF pipelines put y on bottom_top_h (see WRForigPipeline), the others on bottom_top.
    dim_z_y = "bottom_top" if "bottom_top" in ds_y.dims else "bottom_top_h"
    return (ds_y["z"] - ds_y["z_agl"]).isel({dim_z_y: 0})


def add_height_features_wrfpl(ds_X: xr.Dataset, ds_y: xr.Dataset) -> Tuple[Dict[str, xr.DataArray], Dict]:
    """Height features for WRF inputs (X) with native WRF targets (y): ERA5 PL or native levels.

    - `valid`: for ERA5 PL inputs (30 levels), 1 where the PL level lies above the WRF surface.
      step_4c nearest-fills levels with p >= p of the lowest native level; this reproduces that
      criterion exactly. For native inputs (terrain-following, same levels as y) always 1.
    - `lz` in X and y: warped height above ground, the shared interpolation coordinate.
    - `z_sfc`, `lp_sfc`: terrain height [m] and log surface pressure [log Pa].
    """
    dim_z_y = "bottom_top" if "bottom_top" in ds_y.dims else "bottom_top_h"
    p_sfc = ds_y["p"].isel({dim_z_y: 0})  # lowest native level
    dim_z_X = "bottom_top_h"
    n_X = ds_X.sizes[dim_z_X]
    if n_X == len(P_LEVELS_ERA5):
        p_nom = xr.DataArray(P_LEVELS_ERA5, dims=dim_z_X)
        valid = p_nom < p_sfc
    elif n_X == ds_y.sizes[dim_z_y]:
        valid = xr.ones_like(ds_X["z_agl"], dtype=bool)
    else:
        raise ValueError(f"Expected {len(P_LEVELS_ERA5)} ERA5 PL or {ds_y.sizes[dim_z_y]} native input levels, got {n_X}.")
    X_fe = {
        "valid": valid.astype("float32").transpose("sample", dim_z_X),
        "lz": _lz(ds_X["z_agl"]),
        "z_sfc": _z_sfc_wrf(ds_y),
        "lp_sfc": np.log(p_sfc),
    }
    return X_fe, {"lz": _lz(ds_y["z_agl"])}


def add_height_features_era5pl(ds_X: xr.Dataset, ds_y: xr.Dataset) -> Tuple[Dict[str, xr.DataArray], Dict]:
    """Height features for real ERA5 PL (X) with native WRF targets (y). See add_height_features_wrfpl.

    `valid` and `lz` refer to ERA5's own surface (what the input profile itself knows), `z_sfc` to the true
    terrain of the target column, as for WRF PL inputs.
    """
    X_fe = {
        "valid": (ds_X["p"] < ds_X["p_sfc"]).astype("float32").transpose("sample", "bottom_top"),
        "lz": _lz(ds_X["z_agl"]),
        "lp_sfc": np.log(ds_X["p_sfc"]),
        # Drop WRF lat/lon coords: they differ slightly from ERA5's and would conflict on merge
        "z_sfc": _z_sfc_wrf(ds_y).reset_coords(drop=True),
    }
    return X_fe, {"lz": _lz(ds_y["z_agl"])}


# `valid` (exactly 0/1) and `lz` (shared X/y coordinate) are intentionally unscaled, see above.
HEIGHT_SCALER_CONFIG = {
    "z_sfc": (None, "minmax", None),
    "lp_sfc": (None, "minmax", None),
    "p_sfc": (None, "minmax", None),
}
