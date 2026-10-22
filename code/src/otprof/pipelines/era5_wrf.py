import warnings
from typing import List, Literal

import numpy as np
import xarray as xr

from otprof.logging import get_logger
from otprof.pipelines.base import BaseDataPipeline
from otprof.pipelines.shared import (
    ERA5_SCALER_CONFIG,
    ERA5_VARS_PL,
    HEIGHT_SCALER_CONFIG,
    WRF_SCALER_CONFIG,
    WRF_TF_CONFIG,
    add_height_features_era5pl,
    add_X_time_features,
)

logger = get_logger()


class ERA5WRFPipeline(BaseDataPipeline):
    """Original ERA5 PL as input, original WRF as output."""

    mode: Literal["pl"]

    def load_data(self) -> None:
        # Load profiles and surface data
        logger.info(f"Using mode {self.mode.upper()} for vertical levels.")
        ds_era5_vert = xr.load_dataset(self.data_root / f"3_features/era5_{self.mode}.nc")
        ds_era5_sfc = xr.load_dataset(self.data_root / "3_features/era5_sfc.nc")
        ds_era5 = xr.merge([ds_era5_vert, ds_era5_sfc], compat="override")

        # Discard very high levels not present in WRF using precomputed mask
        ds_era5 = ds_era5.sel({"bottom_top": ds_era5["_wrf_mask"], f"bottom_top_h": ds_era5["_wrf_mask_h"]})
        ds_era5 = ds_era5.drop_vars(["_wrf_mask", "_wrf_mask_h"])  # drop masks
        ds_era5 = ds_era5.isel(bottom_top=slice(0, -1))  # Remove highest level to match half-levels

        # Rename half-levels to full-levels for consistency
        bt_h_vars = [v for v in ds_era5.data_vars if "bottom_top_h" in ds_era5[v].dims]
        ds_era5[bt_h_vars] = ds_era5[bt_h_vars].rename(bottom_top_h="bottom_top")
        logger.info(f"Using {ds_era5.sizes['bottom_top']} vertical levels from ERA5 after masking.")

        # Load WRF
        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        # ERA5 alignment done above when discarding too high levels.
        ds_wrf = xr.load_dataset(self.data_root / "3_features/wrf_filtered.nc")
        ds_wrf = ds_wrf.isel(bottom_top=slice(0, -1))

        # Rename half-levels for consistency
        bt_h_vars = [v for v in ds_wrf.data_vars if "bottom_top_h" in ds_wrf[v].dims]
        ds_wrf[bt_h_vars] = ds_wrf[bt_h_vars].rename(bottom_top_h="bottom_top")

        if "lcn2" in self.vars_y:
            # Compute before flattening to avoid MultiIndex discard error
            with warnings.catch_warnings():
                # Catch divide by zero warnings because I will discard later
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ds_wrf["lcn2"] = np.log10(ds_wrf["cn2"])

        # Flatten
        ds_era5 = ds_era5.stack(sample=("time", "loc")).transpose("sample", ...)
        ds_wrf = ds_wrf.stack(sample=("time", "loc")).transpose("sample", ...)

        self.ds_X = ds_era5
        self.ds_y = ds_wrf

    def fill_gaps(self):
        """Filter invalid Cn2 if Cn2 is target."""
        ds_X = self.ds_X
        ds_y = self.ds_y

        if "lcn2" in self.vars_y:
            # Discard invalid Cn2 profiles (NaN from dataset generation)
            cn2_invalid = ds_y["cn2"].isnull().any("bottom_top")
            ds_X = ds_X.sel(sample=~cn2_invalid)
            ds_y = ds_y.sel(sample=~cn2_invalid)

            # Make sure above condition caught all NaNs
            assert (ds_y["lcn2"].isnull().sum() == 0) & (np.isinf(ds_y["lcn2"]).sum() == 0)

        # Save only selected variables
        self.ds_X = ds_X[self.vars_X]
        self.ds_y = ds_y[self.vars_y]


def era5_match_wrf(dp_era5: BaseDataPipeline, dp_wrf: BaseDataPipeline, features: List[str]) -> BaseDataPipeline:
    """Match an ERA5 pipeline onto a WRF pipeline so a WRF-trained model can consume ERA5.

    - Restrict ERA5 inputs to `features`. Levels are already aligned with the WRF-PL grid by
      `ERA5WRFPipeline.load_data` (30 levels, 1000..30 hPa); a level mismatch raises.
    - Adopt the WRF `scaler_X` so ERA5 inputs are normalized in the model's input space.
    """
    dp_era5 = dp_era5.model_copy(deep=True)

    # Ensure the requested features exist on both sides
    features_set = set(features)
    missing_era5 = features_set - set(dp_era5.vars_X)
    if missing_era5:
        raise ValueError(f"ERA5 is missing features: {missing_era5}")
    missing_wrf = features_set - set(dp_wrf.vars_X)
    if missing_wrf:
        raise ValueError(f"WRF is missing features: {missing_wrf}")
    dp_era5.vars_X = list(features)
    dp_era5.ds_X = dp_era5.ds_X[features]

    n_era5 = dp_era5.ds_X.sizes["bottom_top"]
    n_wrf = dp_wrf.ds_X.sizes["bottom_top_h" if "bottom_top_h" in dp_wrf.ds_X.sizes else "bottom_top"]
    if n_era5 != n_wrf:
        raise ValueError(f"ERA5 level count ({n_era5}) does not match WRF input levels ({n_wrf}).")

    # Adopt WRF input scaler (ERA5 now lives in the model's input space)
    dp_era5.scaler_X = dp_wrf.scaler_X
    return dp_era5


def get_era5pl_wrf(domain: str) -> ERA5WRFPipeline:
    """ERA5 PL to WRF original pipeline"""
    p_era5pl_wrf = ERA5WRFPipeline(
        mode="pl",
        domain=domain,
        sel_train=lambda p: {"sample": p.ds_X["time"].dt.year.isin([2019, 2020])},  # 2019-2020
        sel_val=lambda p: {"sample": p.ds_X["time"].dt.year == 2018},  # 2018
        sel_test=lambda p: {"sample": p.ds_X["time"].dt.year == 2017},  # 2017
        vars_X=ERA5_VARS_PL,
        vars_y=[
            "m",
            "u",
            "v",
            "w",
            "d_sin",
            "d_cos",
            "th",
            "p",
            "z_agl",
            "z",
            "lcn2",
            "TSQ",
            "EL_PBL",
            "QKE",
        ],
        fe_fns=[add_X_time_features],
        scaler_config_X=ERA5_SCALER_CONFIG,
        scaler_config_y=WRF_SCALER_CONFIG,
        tf_config_y=WRF_TF_CONFIG,
    )
    return p_era5pl_wrf


def get_era5pl_wrf_h(domain: str) -> ERA5WRFPipeline:
    """ERA5 PL pipeline plus height features for SQF-H (`valid`, `lz`, `z_sfc`, `lp_sfc`).

    `z_sfc` is the true terrain of the target column (WRF), as for the WRF-PL inputs. vars_y are the
    WRF-PL pipeline's pre-feature-engineering targets, so the trained scaler_y covers all of them.
    """
    p = get_era5pl_wrf(domain)
    return p.model_copy(
        update={
            "vars_X": list(ERA5_VARS_PL) + ["p_sfc"],
            "vars_y": ["lcn2", "QKE", "EL_PBL", "TSQ", "z", "z_agl"],
            "fe_fns": p.fe_fns + [add_height_features_era5pl],
            "scaler_config_X": {**p.scaler_config_X, **HEIGHT_SCALER_CONFIG},
        }
    )
