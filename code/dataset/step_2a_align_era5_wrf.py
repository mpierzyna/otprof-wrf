"""Align ERA5 and WRF data spatially, temporally, and ensure overlapping variables have the same names."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import xarray as xr
import pathlib
import warnings
from scipy.spatial import KDTree

from research_tools.reporter import BaseReport

ROOT_RAW = pathlib.Path("1_raw")
ERA5_ROOT = ROOT_RAW / "era5"
WRF_ROOT = ROOT_RAW / "wrf_cn2"

ROOT_ALIGNED = pathlib.Path("2_aligned")
ROOT_ALIGNED.mkdir(exist_ok=True)

ERA5_RENAME = {
    "geopotential": "z",
    "geopotential_at_surface": "z_sfc",
    "potential_vorticity": "pv",
    "vertical_velocity": "w",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "tk",
    "specific_humidity": "q",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "tk2",
    "boundary_layer_height": "blh",
    "surface_pressure": "p_sfc",
    "surface_latent_heat_flux": "lh",
    "mean_surface_net_short_wave_radiation_flux": "msnsrf",
    "low_cloud_cover": "lcc",
    "skin_temperature": "tsk",
    "land_sea_mask": "lsm",
    "mean_sea_level_pressure": "slp",
    # Sfc fluxes
    "friction_velocity": "ust",
    "instantaneous_surface_sensible_heat_flux": "hfx",
    # Coords
    "latitude": "lat",
    "longitude": "lon",
}

WRF_RENAME = {
    "HGT": "z_sfc",
    "u_met": "u",
    "v_met": "v",
    "PBLH": "blh",
    "LANDMASK": "lsm",
    # Sfc vars
    "U10": "u10",
    "V10": "v10",
    "T2": "tk2",
    # Sfc fluxes
    "UST": "ust",
    "HFX": "hfx",
    "LH": "lh",
    # Coords, keep bottom-top because it is different from ERA5's level
    "XLAT": "lat",
    "XLONG": "lon",
}


def plot_lat_lon(ds_era5, ds_wrf) -> plt.Figure:
    # Required to generate a mesh for plotting
    if "loc" not in ds_era5.dims:
        ds_era5 = ds_era5.stack(loc=("lat", "lon"))
    if "loc" not in ds_wrf.dims:
        ds_wrf = ds_wrf.stack(loc=("south_north", "west_east"))

    fig, ax = plt.subplots()
    ax.scatter(ds_era5["lon"].values, ds_era5["lat"].values, s=1, label="ERA5")
    ax.scatter(ds_wrf["lon"].values, ds_wrf["lat"].values, s=1, label="WRF")
    ax.legend()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig


def plot_distance_hist(dist, dist_th) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))

    ax1.hist(np.sqrt(dist), bins=100)
    ax1.axvline(np.sqrt(dist_th), color="r", linestyle="--")
    dist_ticks = np.linspace(0, np.max(dist), 5)
    ax1.set_xticks(np.sqrt(dist_ticks))
    ax1.set_xticklabels([f"{d:.2f}" for d in dist_ticks])

    ax2.hist(dist[dist < dist_th], bins=100)

    return fig


if __name__ == "__main__":

    # Grid params default to the original `nl` values; override per domain from params.yaml via CLI.
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dist-th", type=float, default=0.02)  # ~2km spacing
    _ap.add_argument("--crop", type=int, default=3)  # crop outer cells to avoid edge effects
    _args = _ap.parse_args()
    DIST_TH = _args.dist_th
    CROP = _args.crop

    with BaseReport("Report ERA5 WRF alignment", ROOT_ALIGNED / f"report.html") as r:
        # Load ERA5 data
        print("Loading ERA5 data...", end="")
        ds_era5 = xr.open_mfdataset(str(ERA5_ROOT / "pl" / "*.nc"))
        ds_era5 = ds_era5.rename(ERA5_RENAME)
        # Normalize ERA5 longitude from ARCO's 0..360 convention to WRF's -180..180 so the
        # spatial KDTree alignment matches (no-op for domains with only positive lons, e.g. nl).
        ds_era5 = ds_era5.assign_coords(lon=(((ds_era5["lon"] + 180) % 360) - 180))
        print("Done!")
        r.add_text(f"Original ERA5 shape: {ds_era5.sizes}")

        # Flip sign of fluxes to match WRF convention
        ds_era5["hfx"] = -ds_era5["hfx"]
        ds_era5["lh"] = -ds_era5["lh"] / 3600  # Convert from J/m2 to W/m2 (not needed for HFX)

        # MSL pressure Pa -> hPa
        ds_era5["slp"] = ds_era5["slp"] / 100

        # Load WRF data and align each file directly to keep RAM usage low
        print("Loading WRF data...")
        print(f"Aligning spatially with dist_th={DIST_TH}...")
        wrfout_files = sorted(WRF_ROOT.glob("wrfout_*"))
        r.add_text("Processed WRF files:")
        r.add_list(wrfout_files)

        # Stack ERA5 for horizontal direction for alignment
        ds_era5 = ds_era5.stack(loc=("lat", "lon"))
        with warnings.catch_warnings():
            # Ignore "p" serialization warning about two fill values
            warnings.filterwarnings("ignore", category=xr.SerializationWarning)

            ds_wrf_list = []
            dist, idx, dist_mask = None, None, None
            for f in tqdm.tqdm(wrfout_files):
                ds_wrf_i = xr.load_dataset(f)
                ds_wrf_i = ds_wrf_i.isel(
                    south_north=slice(CROP, -CROP), west_east=slice(CROP, -CROP)
                )  # crop outer-most cells bc of edge effects
                ds_wrf_i = ds_wrf_i.stack(loc=("south_north", "west_east"))
                ds_wrf_i = ds_wrf_i.rename(WRF_RENAME)

                # WRF grid doesn't move in time, so we compute alignment on first file and reuse it
                if dist_mask is None:
                    r.add_text(f"Original WRF shape: {ds_wrf_i.sizes}")
                    r.add_mpl_fig(plot_lat_lon(ds_era5, ds_wrf_i), "Grid points before alignment")

                    tree = KDTree(np.column_stack((ds_wrf_i["lat"].values, ds_wrf_i["lon"].values)))
                    dist, idx = tree.query(np.column_stack((ds_era5["lat"].values, ds_era5["lon"].values)), k=1)
                    dist_mask = dist < DIST_TH
                    r.add_mpl_fig(plot_distance_hist(dist, DIST_TH), "Distance histogram with threshold")

                ds_wrf_i = ds_wrf_i.isel(loc=idx[dist_mask])
                ds_wrf_list.append(ds_wrf_i)

            ds_wrf = xr.concat(ds_wrf_list, dim="time")

        # Select only locations of ERA5 paired with WRF.
        ds_era5 = ds_era5.isel(loc=dist_mask)

        print(ds_era5.sizes, ds_wrf.sizes)
        r.add_text(f"ERA5 / WRF shapes after spatial alignment: {ds_era5.sizes}, {ds_wrf.sizes}")
        r.add_mpl_fig(plot_lat_lon(ds_era5, ds_wrf), "Grid points after alignment")

        # # Align temporally
        print("Aligning time dimensions...")
        time_overlap = ds_era5.time.isin(ds_wrf.time)
        ds_era5 = ds_era5.sel(time=time_overlap)
        ds_wrf = ds_wrf.sel(time=~ds_wrf.time.isnull())  # some NaT... Don't know why.
        ds_wrf = ds_wrf.sel(time=ds_era5.time)
        print(ds_era5.sizes, ds_wrf.sizes)
        r.add_text(f"ERA5 / WRF shapes after temporal alignment: {ds_era5.sizes}, {ds_wrf.sizes}")

        # Reset stacked indices because they cannot be saved
        ds_era5 = ds_era5.reset_index("loc")
        ds_wrf = ds_wrf.reset_index("loc")

        # Split ERA5 into pl and sfc datasets
        vars_pl = [v for v in ds_era5.data_vars if "level" in ds_era5[v].dims]
        ds_era5_pl = ds_era5[vars_pl]
        vars_sfc = [v for v in ds_era5.data_vars if v not in vars_pl]
        ds_era5_sfc = ds_era5[vars_sfc]

        # Flip ERA5 PL vertical coordinate to have consistent bottom-top ordering with WRF
        ds_era5_pl = ds_era5_pl.isel(level=slice(None, None, -1))
        ds_era5_pl = ds_era5_pl.rename({"level": "bottom_top"})

        # Keep levels as separate variable not as coordinate
        level = ds_era5_pl["bottom_top"].drop_vars("bottom_top")
        ds_era5_pl = ds_era5_pl.drop_vars("bottom_top")
        ds_era5_pl["level"] = level

        # Save aligned datasets
        print("Writing to disk...")
        ds_era5_pl.to_netcdf(ROOT_ALIGNED / f"era5_pl_aligned.nc", engine="netcdf4")
        ds_era5_sfc.to_netcdf(ROOT_ALIGNED / f"era5_sfc_aligned.nc", engine="netcdf4")
        ds_wrf.to_netcdf(ROOT_ALIGNED / f"wrf_aligned.nc", engine="netcdf4")
        print("Done!")
