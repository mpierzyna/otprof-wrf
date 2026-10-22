from typing import Dict, Tuple
import xarray as xr
import numpy as np
import pathlib

from utils import get_half_levels
from research_tools.ot import ot, most


ROOT_ALIGNED = pathlib.Path("2_aligned")
ROOT_FEATURES = pathlib.Path("3_features")
ROOT_FEATURES.mkdir(exist_ok=True)


def compute_grads(ds: xr.Dataset, dim: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute wind shear and pot temp grad along given dimension."""
    # Layer thickness
    dz = ds["z"].diff(dim)

    # Compute wind shear
    du = ds["u"].diff(dim)
    dv = ds["v"].diff(dim)
    du_dz = du / dz
    dv_dz = dv / dz
    S = np.sqrt(du_dz**2 + dv_dz**2)
    S = S.rename({dim: f"{dim}_h"})
    S = S.astype(np.float32)
    S = S.assign_attrs(long_name="Wind shear", units="1/s")

    # Compute potential temperature and its gradient
    dth = ds["th"].diff(dim)
    dth_dz = dth / dz
    dth_dz = dth_dz.rename({dim: f"{dim}_h"})
    dth_dz = dth_dz.astype(np.float32)
    dth_dz = dth_dz.assign_attrs(long_name="Vertical gradient of potential temperature", units="K/m")

    # If dim has a coordinate, compute half-level location/coordinates and assign to gradients
    if dim in ds.coords:
        coord_h = get_half_levels(ds[dim], dim)
        coord_h = coord_h.astype(np.float32)
        S = S.assign_coords({f"{dim}_h": coord_h})
        dth_dz = dth_dz.assign_coords({f"{dim}_h": coord_h})

    return S, dth_dz


def compute_m_d(ds: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute wind magnitude and direction"""
    u = ds["u"]
    v = ds["v"]
    m = np.sqrt(u**2 + v**2)
    m = m.astype(np.float32)
    m = m.assign_attrs(long_name="Wind magnitude", units="m/s")

    d = np.rad2deg(np.arctan2(-u, -v))
    d = d.astype(np.float32)
    d = d.assign_attrs(long_name="Horizontal wind direction", units="deg")

    d_sin = np.sin(np.deg2rad(d))
    d_cos = np.cos(np.deg2rad(d))
    d_sin = d_sin.astype(np.float32)
    d_sin = d_sin.assign_attrs(long_name="Sine of horizontal wind direction", units="1")
    d_cos = d_cos.astype(np.float32)
    d_cos = d_cos.assign_attrs(long_name="Cosine of horizontal wind direction", units="1")

    return m, d, d_sin, d_cos


def compute_sfc_ct2(
    u_st: xr.DataArray,
    shfx: xr.DataArray,
    T_mean: xr.DataArray,
    z_agl: xr.DataArray | float,
    z_dim: str | None,
) -> xr.DataArray:
    """Estimate CT2 at the surface using the W71 flux model.

    - If z_dim is None, evaluate at constant height z_agl (float).
    - If z_dim is given, evaluate at the lowest model level (z_agl) along z_dim.
    """
    # Obukhov length
    k = 0.4
    L = -(u_st**3) * T_mean / (k * shfx * 9.81)

    # Characteristic temperature scale
    theta_st = -shfx / u_st

    if z_dim is None:
        # Evaluate similarity function at constant height
        assert np.isscalar(z_agl), "z_agl must be scalar if z_dim is None"
        z_bot = z_agl
    else:
        # Evaluate similarity function at lowest model level
        # Make sure we have bottom_top vertical ordering
        z_bot, z_top = z_agl.isel({z_dim: 0}), z_agl.isel({z_dim: -1})
        if (z_bot > z_top).all():
            z_agl = z_agl.isel({z_dim: slice(None, None, -1)})
            z_bot, z_top = z_top, z_bot

        # ERA5 PL has z_agl < 0 because 1000 hPa in terrain is below surface pressure.
        # Select first non-zero level
        if (z_bot <= 0).any():
            z_above_sfc = z_agl >= 0
            i_above_sfc = xr.apply_ufunc(
                np.searchsorted,
                z_above_sfc,
                True,
                input_core_dims=[[z_dim], []],
                vectorize=True,
            )
            z_bot = z_agl.isel({z_dim: i_above_sfc})

    ct2_w71f = most.w71_flux_ct2(T_st=theta_st, z=z_bot, L=L)
    ct2_w71f = ct2_w71f.clip(1e-8, None)  # avoid zero or negative Cn2
    ct2_w71f = ct2_w71f.assign_attrs(
        long_name="Temperature structure parameter CT2 estimated with W71 surface flux model",
        units="K^2 m^(-2/3)",
    )

    return ct2_w71f


def compute_astroclim_targets(
    cn2: xr.DataArray,
    m: xr.DataArray,
    z_agl: xr.DataArray,
    blh: xr.DataArray,
) -> Dict[str, xr.DataArray]:
    """Compute astroclimate targets."""
    # Compute region masks
    # SL and BL masks need to be extended one element to the right for piecewise integration
    region_masks = {
        "sl": ot._extend_mask_right(z_agl <= (blh * 0.1), dim="bottom_top"),  # surface layer
        "bl": ot._extend_mask_right(z_agl <= blh, dim="bottom_top"),  # entire boundary layer
        "fa": z_agl > blh,  # free atmosphere
        "full": np.ones_like(z_agl, dtype=bool),  # full profile
    }
    regions_long = {
        "sl": "surface layer",
        "bl": "boundary layer",
        "fa": "free atmosphere",
        "full": "full column",
    }

    features = {}
    for r, mask in region_masks.items():

        r0 = ot.fried_r0_xr(cn2=cn2.where(mask), z=z_agl.where(mask), dim="bottom_top")
        r0 = r0.where(~np.isinf(r0))  # integration with too few levels can yield inf
        r0 = r0.astype(np.float32).assign_attrs(
            long_name=f"Fried parameter ({regions_long[r]})",
            units="m",
        )
        features[f"r0_{r}"] = r0

        tau0 = ot.wave_coherence_time_xr(cn2=cn2.where(mask), m=m.where(mask), z=z_agl.where(mask), dim="bottom_top")
        tau0 = tau0.where(~np.isinf(tau0))  # integration with too few levels can yield inf
        tau0 = tau0.astype(np.float32).assign_attrs(
            long_name=f"Wavefront coherence time ({regions_long[r]})",
            units="s",
        )
        features[f"tau0_{r}"] = tau0

        th0 = ot.isoplanatic_angle_xr(cn2=cn2.where(mask), z=z_agl.where(mask), dim="bottom_top")
        th0 = th0.where(~np.isinf(th0))  # integration with too few levels can yield inf
        th0 = th0.astype(np.float32).assign_attrs(
            long_name=f"Isoplanatic angle ({regions_long[r]})",
            units="rad",
        )
        features[f"th0_{r}"] = th0

        lsi2 = np.log10(ot.scint_index_xr(cn2=cn2.where(mask), z=z_agl.where(mask), dim="bottom_top", mode="plane"))
        lsi2 = lsi2.where(~np.isinf(lsi2))  # integration with too few levels can yield inf
        lsi2 = lsi2.astype(np.float32).assign_attrs(
            long_name=f"Log10 scintillation index ({regions_long[r]})",
            units="1",
        )
        features[f"lsi2_{r}"] = lsi2

    return features


def compute_era5_features(ds_era5: xr.Dataset, ds_era5_sfc: xr.Dataset, z_dim: str) -> xr.Dataset:
    """Compute features for ERA5 pressure and model level data."""
    # Convert geopotential to height in m
    print("-> Making features...")
    ds_era5["z"] = (ds_era5["z"] / 9.81).assign_attrs(
        long_name="Height above MSL",
        units="m",
    )
    ds_era5["z_agl"] = (ds_era5["z"] - ds_era5_sfc["z_sfc"]).assign_attrs(
        long_name="Height above ground (AGL)",
        units="m",
    )
    ds_era5["z_rel"] = (ds_era5["z_agl"] / ds_era5_sfc["blh"]).assign_attrs(
        long_name="Height above ground (AGL) normalized by BLH",
        units="1",
    )

    # Add potential temperature
    p0 = 100000  # Pa
    gamma = 1.4
    th = ds_era5["tk"] * (p0 / ds_era5["p"]) ** ((gamma - 1) / gamma)
    th = th.astype(np.float32)
    th = th.assign_attrs(long_name="Potential temperature", units="K")
    ds_era5["th"] = th

    # Add gradients
    S, dth_dz = compute_grads(ds_era5, z_dim)
    ds_era5["S"] = S
    ds_era5["dth_dz"] = dth_dz

    # Add wind magnitude and direction
    m, d, d_sin, d_cos = compute_m_d(ds_era5)
    ds_era5["m"] = m
    ds_era5["d"] = d
    ds_era5["d_sin"] = d_sin
    ds_era5["d_cos"] = d_cos

    return ds_era5


def compute_wrf_features(ds_wrf: xr.Dataset) -> xr.Dataset:
    """Compute features for WRF data."""
    S, dth_dz = compute_grads(ds_wrf, "bottom_top")
    ds_wrf["S"] = S
    ds_wrf["dth_dz"] = dth_dz
    ds_wrf["p_h"] = get_half_levels(ds_wrf["p"], "bottom_top")  # compute half-level pressure
    ds_wrf["z_h"] = get_half_levels(ds_wrf["z"], "bottom_top")  # compute half-level height

    m, d, d_sin, d_cos = compute_m_d(ds_wrf)
    ds_wrf["m"] = m
    ds_wrf["d"] = d
    ds_wrf["d_sin"] = d_sin
    ds_wrf["d_cos"] = d_cos

    z_agl = ds_wrf["z"] - ds_wrf["z_sfc"]
    ds_wrf["z_rel"] = (z_agl / ds_wrf["blh"]).assign_attrs(
        long_name="Height above ground (AGL) normalized by BLH",
        units="1",
    )
    ds_wrf["z_agl"] = z_agl.assign_attrs(
        long_name="Height above ground (AGL)",
        units="m",
    )

    # Add CT2 estimates
    ct2_w71f = compute_sfc_ct2(
        u_st=ds_wrf["ust"],
        shfx=ds_wrf["hfx"] / 1.216e3,  # (W/m2 -> K m/s)
        T_mean=ds_wrf["tk2"],
        z_agl=2.0,  # evaluate at 2m AGL constant height to match WRF and ERA5
        z_dim=None,
    )
    ds_wrf["lct2_w71f"] = np.log10(ct2_w71f)

    return ds_wrf


if __name__ == "__main__":
    ## Load ERA5 surface data
    print("Loading ERA5 surface data...")
    ds_era5_sfc = xr.load_dataset(ROOT_ALIGNED / f"era5_sfc_aligned.nc")
    ds_era5_sfc["z_sfc"] = ds_era5_sfc["z_sfc"].mean("time")  # varies little over time (see eda_aligned.ipynb)
    ds_era5_sfc["z_sfc"] = (ds_era5_sfc["z_sfc"] / 9.81).assign_attrs(long_name="Terrain height", units="m")

    # Add CT2 surface estimate
    ct2_w71f = compute_sfc_ct2(
        u_st=ds_era5_sfc["ust"],
        shfx=ds_era5_sfc["hfx"] / 1.216e3,  # (W/m2 -> K m/s)
        T_mean=ds_era5_sfc["tk2"],
        z_agl=2.0,  # evaluate at 2m AGL constant height to match WRF and ERA5
        z_dim=None,
    )
    ds_era5_sfc["lct2_w71f"] = np.log10(ct2_w71f)

    ## Load ERA5 pressure level data
    print("Loading ERA5 pressure data...")
    ds_era5_pl = xr.load_dataset(ROOT_ALIGNED / f"era5_pl_aligned.nc")

    # Add pressure in Pa and compute features
    pl = ds_era5_pl["level"]
    ds_era5_pl["p"] = (pl * 100).astype(int).assign_attrs(long_name="Pressure", units="Pa")
    ds_era5_pl = compute_era5_features(ds_era5=ds_era5_pl, ds_era5_sfc=ds_era5_sfc, z_dim="bottom_top")

    # Compute masks for levels with p > 1000 Pa to match WRF
    p_mask = ds_era5_pl["p"] > 1000
    ds_era5_pl["_wrf_mask"] = p_mask
    ds_era5_pl["_wrf_mask_h"] = p_mask[1:].rename({"bottom_top": "bottom_top_h"}).astype(bool)

    ## Load ERA5 model level data
    # print("Loading ERA5 model level data...")
    # ds_era5_ml = xr.load_dataset(ROOT_ALIGNED / f"era5_ml_aligned.nc")
    # ds_era5_ml = compute_era5_features(ds_era5=ds_era5_ml, ds_era5_sfc=ds_era5_sfc, z_dim="bottom_top")

    # # Compute masks for levels with p > 1000 Pa to match WRF
    # p_mask = (ds_era5_ml["p"] > 1000).any(["time", "loc"])
    # ds_era5_ml["_wrf_mask"] = p_mask
    # ds_era5_ml["_wrf_mask_h"] = p_mask[1:].rename({"bottom_top": "bottom_top_h"}).astype(bool)

    ## Load WRF and compute gradients
    print("Loading WRF data...")
    ds_wrf = xr.load_dataset(ROOT_ALIGNED / f"wrf_aligned.nc")

    print("-> Making features...")
    ds_wrf = compute_wrf_features(ds_wrf)

    # Compute astroclimate parameters (additional targets)
    print("-> Computing astroclimate parameters...")
    astroclim_targets = compute_astroclim_targets(
        cn2=ds_wrf["cn2"],
        m=ds_wrf["m"],
        z_agl=ds_wrf["z_agl"],
        blh=ds_wrf["blh"],
    )
    for k, v in astroclim_targets.items():
        ds_wrf[k] = v

    # Write to file
    print("Writing all datasets to disk...")
    ds_era5_sfc.to_netcdf(ROOT_FEATURES / f"era5_sfc.nc", engine="netcdf4")
    ds_era5_pl.to_netcdf(ROOT_FEATURES / f"era5_pl.nc", engine="netcdf4")
    # ds_era5_ml.to_netcdf(ROOT_FEATURES / f"era5_ml.nc", engine="netcdf4")
    ds_wrf.to_netcdf(ROOT_FEATURES / f"wrf_all.nc", engine="netcdf4")
    print("Done!")
