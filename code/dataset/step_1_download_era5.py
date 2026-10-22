#!/usr/bin/env python3
"""Download ERA5 pressure-level and surface data at the WRF Cn2 time steps from the ARCO ERA5 store
(Google Cloud, anonymous access) into 1_raw/era5/pl/. Run from inside dataset/data/<domain>/ (see dvc.yaml).
An existing output file is not downloaded again.
"""
import argparse
from typing import List

import pathlib
import xarray as xr
import tqdm
import warnings

# Root for ERA5 and WRF raw data
RAW_ROOT = pathlib.Path("1_raw")


def parse_area() -> dict:
    """Parse the ERA5 spatial subset from CLI. Defaults match the original `nl` domain.

    ERA5 latitude is stored in descending order, so the latitude slice runs from
    lat_max down to lat_min.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--lon-min", type=float, default=2)
    p.add_argument("--lon-max", type=float, default=9)
    p.add_argument("--lat-max", type=float, default=55)
    p.add_argument("--lat-min", type=float, default=49)
    a = p.parse_args()
    return {"longitude": slice(a.lon_min, a.lon_max), "latitude": slice(a.lat_max, a.lat_min)}

# WRF Cn2 data location
WRF_RAW = RAW_ROOT / "wrf_cn2"
wrf_files = sorted(WRF_RAW.glob("*cn2.nc"))
assert WRF_RAW.exists(), "WRF raw data not found. Check if you execute this script from data directory."


def download_era5_sl_pl(vars: List[str], out_file: str, area: dict):
    """Download ERA5 surface and pressure level data matching WRF Cn2 data."""
    # Check if output file exists
    out_file = RAW_ROOT / "era5" / out_file
    if out_file.exists():
        print(f"Output file {out_file} exists already. Skipping download.")
        return

    # Check if directory exists
    out_file.parent.mkdir(exist_ok=True, parents=True)

    # Open all datasets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Catch "variable 'p' has multiple fill values"
        wrf_ds_list = [xr.open_dataset(f) for f in tqdm.tqdm(wrf_files)]

    # Concatenate time dimension
    wrf_time = xr.concat([ds.time for ds in wrf_ds_list], dim="time")
    wrf_time = wrf_time[~wrf_time.isnull()]
    print(f"{len(wrf_time)} time steps in WRF data.")

    # Open ERA5 cloud dataset
    print("Opening and subsetting ERA5 data...")
    ds_era5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )

    # Subset time and variables. No spatial chunks, so we always need to download globally
    ds_era5 = ds_era5.sel(time=wrf_time)[vars]
    size_download_gb = ds_era5.nbytes / 1e9
    print(f"Size to download: {size_download_gb:.1f} GB, ~{size_download_gb/0.100/(60 * 60):.1f} hours at 100 Mbit/s")

    # But only store area of interests -> spatial subset
    ds_era5 = ds_era5.sel(area)
    print(f"Size on disk: {ds_era5.nbytes / 1e9:.1f} GB")
    print(ds_era5)

    # Save to file
    print(f"Saving to {out_file}. This may take a while.")
    ds_era5.to_netcdf(out_file)
    print("Done.")


if __name__ == "__main__":
    area = parse_area()

    download_era5_sl_pl(
        [
            # Profile
            "geopotential",
            "potential_vorticity",
            "vertical_velocity",
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            # Surface
            "soil_type",
            "land_sea_mask",
            "angle_of_sub_gridscale_orography",
            "anisotropy_of_sub_gridscale_orography",
            "geopotential_at_surface",
            "boundary_layer_height",
            "boundary_layer_dissipation",
            "surface_pressure",
            "friction_velocity",
            "instantaneous_surface_sensible_heat_flux",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_latent_heat_flux",
            "mean_surface_net_short_wave_radiation_flux",
            "low_cloud_cover",
            "skin_temperature",
        ],
        out_file="pl/era5_001.nc",
        area=area,
    )
