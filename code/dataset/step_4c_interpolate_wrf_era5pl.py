"""Interpolate WRF to ERA5 pressure levels."""

import pathlib

import numpy as np
import xarray as xr
from step_3a_make_features import compute_wrf_features
from utils import interp_vert

ROOT_FEATURES = pathlib.Path("3_features")
ROOT_FEATURES_INTERP = pathlib.Path("4_features_interp")
ROOT_FEATURES_INTERP.mkdir(exist_ok=True, parents=True)


# Load WRF data to memory for fast computation
print("Loading WRF data...")
ds_wrf = xr.load_dataset(ROOT_FEATURES / "wrf_filtered.nc")

# Drop features, which will be recomputed later
ds_wrf = ds_wrf.drop_vars(["m", "d", "d_sin", "d_cos", "S", "dth_dz", "p_h", "z_h", "z_agl", "z_rel"])

# Invert vertical levels bc interpolation requires increasing pressure (top to bottom)
ds_wrf = ds_wrf.isel(bottom_top=slice(None, None, -1))

# Interpolate entire dataset to ERA5 pressure levels
print("Interpolating WRF data to ERA5 pressure levels...")
p_hPa_target = xr.DataArray(
    [
        # WRF doesn't go that high
        # 1,
        # 2,
        # 3,
        # 5,
        # 7,
        # 10,
        20,
        30,
        50,
        70,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    ],
    dims="bottom_top",
)

# Interpolate on log-pressure scale like ECMWF example
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Step2Interpolatevariablesonmodellevelstocustompressurelevels
p_hPa = ds_wrf["p"] / 100.0  # convert to hPa
ds_wrf_interp = interp_vert(ds_wrf, np.log(p_hPa), np.log(p_hPa_target), "bottom_top")

# lowest level can be NaN if WRF sfc pressure < 1000 hPa -> fill by nearest
ds_wrf_interp = ds_wrf_interp.interpolate_na(dim="bottom_top", method="nearest", fill_value="extrapolate")

# Add surface variables back
v_sfc = [v for v in ds_wrf.data_vars if v not in list(ds_wrf_interp.data_vars)]
ds_wrf_interp = ds_wrf_interp.merge(ds_wrf[v_sfc])

# For consistency, we compute features again.
# In true ERA5 PL, gradients would also be computed on lower resolution.
print("Recomputing WRF features on interpolated data...")
ds_wrf_interp = compute_wrf_features(ds_wrf_interp)

# Interpolate features again. Above nearest extrapolation will create dz=0 for gradients at surface.
ds_wrf_interp = ds_wrf_interp.interpolate_na(dim="bottom_top_h", method="nearest", fill_value="extrapolate")

# Flip levels back to original order (bottom to top)
ds_wrf_interp = ds_wrf_interp.isel(bottom_top=slice(None, None, -1), bottom_top_h=slice(None, None, -1))

# Save interpolated data
print("Writing to disk...")
ds_wrf_interp.to_netcdf(ROOT_FEATURES_INTERP / "wrf_filtered_era5pl.nc")
print("Done!")
