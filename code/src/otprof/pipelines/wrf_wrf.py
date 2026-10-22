import warnings
from typing import Literal

import numpy as np
import xarray as xr

from otprof.pipelines.base import BaseDataPipeline
from otprof.pipelines.shared import (
    HEIGHT_SCALER_CONFIG,
    WRF_SCALER_CONFIG,
    WRF_TF_CONFIG,
    add_height_features_wrfpl,
    add_X_time_features,
)


class WRForigPipeline(BaseDataPipeline):
    """WRF data without modification as input and output"""

    def load_data(self):
        # Load
        ds_wrf = xr.load_dataset(self.data_root / "3_features/wrf_filtered.nc")

        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        bt_full_vars = [v for v in ds_wrf.data_vars if "bottom_top" in ds_wrf[v].dims]
        ds_wrf[bt_full_vars] = ds_wrf[bt_full_vars].isel(bottom_top=slice(0, -1)).rename(bottom_top="bottom_top_h")

        # Compute before flattening to avoid MultiIndex discard error
        with warnings.catch_warnings():
            # Catch divide by zero warnings because I will discard later
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_wrf["lcn2"] = np.log10(ds_wrf["cn2"])

        # Flatten
        ds_wrf = ds_wrf.stack(sample=("time", "loc")).transpose("sample", ...)

        # Discard invalid Cn2 profiles (NaN from dataset generation)
        cn2_invalid = ds_wrf["cn2"].isnull().any("bottom_top_h")
        ds_wrf = ds_wrf.sel(sample=~cn2_invalid)

        # Make sure above condition caught all NaNs
        assert (ds_wrf["lcn2"].isnull().sum() == 0) & (np.isinf(ds_wrf["lcn2"]).sum() == 0)

        self.ds_X = ds_wrf
        self.ds_y = ds_wrf


class WRFWRFPipeline(BaseDataPipeline):
    """WRF interpolated to ERA5 PL as input, original WRF as output"""

    mode: Literal["era5pl"]

    def load_data(self):
        # Load
        ds_wrf_X = xr.load_dataset(self.data_root / f"4_features_interp/wrf_filtered_{self.mode}.nc")
        ds_wrf_X = ds_wrf_X.drop_vars(["bottom_top", "bottom_top_h"])  # drop coordinates
        ds_wrf_y = xr.load_dataset(self.data_root / "3_features/wrf_filtered.nc")  # standard bottom_top

        # For compatability with WRFonly pipeline, remove highest y levels
        ds_wrf_y = ds_wrf_y.isel(bottom_top=slice(0, -1), bottom_top_h=slice(0, -1))

        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        full_vars = [v for v in ds_wrf_X.data_vars if "bottom_top" in ds_wrf_X[v].dims]
        ds_wrf_X[full_vars] = ds_wrf_X[full_vars].isel(bottom_top=slice(0, -1)).rename(bottom_top="bottom_top_h")

        # Compute before flattening to avoid MultiIndex discard error
        with warnings.catch_warnings():
            # Catch divide by zero warnings because I will discard later
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_wrf_y["lcn2"] = np.log10(ds_wrf_y["cn2"])

        # Flatten
        ds_wrf_X = ds_wrf_X.stack(sample=("time", "loc")).transpose("sample", ...)
        ds_wrf_y = ds_wrf_y.stack(sample=("time", "loc")).transpose("sample", ...)

        # Discard invalid Cn2 profiles (NaN from dataset generation)
        cn2_invalid = ds_wrf_y["cn2"].isnull().any("bottom_top")
        ds_wrf_X = ds_wrf_X.sel(sample=~cn2_invalid)
        ds_wrf_y = ds_wrf_y.sel(sample=~cn2_invalid)

        # Make sure above condition caught all NaNs
        assert (ds_wrf_y["lcn2"].isnull().sum() == 0) & (np.isinf(ds_wrf_y["lcn2"]).sum() == 0)
        self.ds_X = ds_wrf_X
        self.ds_y = ds_wrf_y


def get_wrf_orig(domain: str) -> WRForigPipeline:
    """Get WRF original in and out pipeline for a given domain."""
    p_wrf_orig = WRForigPipeline(
        domain=domain,
        sel_train=lambda p: {"sample": p.ds_X["time"].dt.year.isin([2019, 2020])},  # 2019-2020
        sel_val=lambda p: {"sample": p.ds_X["time"].dt.year == 2018},  # 2018
        sel_test=lambda p: {"sample": p.ds_X["time"].dt.year == 2017},  # 2017
        vars_X=[
            # "z_rel",
            "z",
            "z_agl",
            "m",
            "d_sin",
            "d_cos",
            "p",
            "u",
            "v",
            "w",
            "th",
            "S",
            "dth_dz",
            # Surface
            "ust",
            "hfx",
            "lh",
            "blh",
            "u10",
            "v10",
            "tk2",
            "slp",
            "lsm",
            "lct2_w71f",
        ],
        vars_y=[
            "lcn2",
            "QKE",
            "EL_PBL",
            "TSQ",
            "z",
            "z_agl",
        ],
        fe_fns=[add_X_time_features],
        scaler_config_X=WRF_SCALER_CONFIG,
        scaler_config_y=WRF_SCALER_CONFIG,
        tf_config_y=WRF_TF_CONFIG,
    )
    return p_wrf_orig


def get_wrf_orig_h(domain: str) -> WRForigPipeline:
    """WRF native pipeline plus height features for SQF-H (all levels valid, identity interpolation)."""
    p = get_wrf_orig(domain)
    return p.model_copy(
        update={
            "fe_fns": p.fe_fns + [add_height_features_wrfpl],
            "scaler_config_X": {**p.scaler_config_X, **HEIGHT_SCALER_CONFIG},
        }
    )


def get_wrfpl_wrf(domain: str) -> WRFWRFPipeline:
    """WRF input to ERA5 PL, same settings as original WRF"""
    p_wrf_orig = get_wrf_orig(domain)
    p_wrfpl_wrf = WRFWRFPipeline(mode="era5pl", **p_wrf_orig.model_dump())
    return p_wrfpl_wrf


def get_wrfpl_wrf_h(domain: str) -> WRFWRFPipeline:
    """WRF-PL pipeline plus height features (`valid`, `lz`, `z_sfc`, `lp_sfc`) for SQF-H."""
    p = get_wrfpl_wrf(domain)
    return p.model_copy(
        update={
            "fe_fns": p.fe_fns + [add_height_features_wrfpl],
            "scaler_config_X": {**p.scaler_config_X, **HEIGHT_SCALER_CONFIG},
        }
    )
