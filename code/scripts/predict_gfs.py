"""SQF-H Cn2 profiles from GFS forecasts for locations of your choice (post-processing of GFS).

Downloads only the needed fields of one GFS forecast cycle from the NOAA GFS archive on AWS (anonymous, byte ranges
from the .idx inventories), converts them to the conventions of the ERA5 training inputs (otprof.predict.make_pl_inputs)
and predicts with the best checkpoint of the decoupled model (wrf_pl) at the heights Z_AGL above the GFS ground.
Optionally applies the nl output quantile mapping (nl-oQM, fitted on ERA5-driven predictions) to lcn2 and computes r0
and lsi2. Needs runs/models/wrf_pl/ (trained model with oqm_lcn2.joblib).

GFS files and fields (0.25 deg grid unless noted):
- pgrb2:  HGT, TMP, UGRD, VGRD on 25 of the 31 pressure levels 1000..20 hPa; surface pressure, orography, land mask,
          2 m temperature, 10 m wind, mean sea level pressure.
- pgrb2b: the six remaining levels 875, 825, 775, 225, 175, 125 hPa.
- sflux (Gaussian grid, ~13 km, nearest point): instantaneous sensible and latent heat flux (upward positive like the
          training inputs, so not sign-flipped as ERA5), friction velocity, PBL height.
About 160 fields of 0.5-4 MB per lead time. The AWS archive starts in 2021.

Differences to the paper setup: the inputs come from GFS instead of ERA5 (different model physics, in particular for
the PBL height, friction velocity and surface fluxes), the terrain height input is GFS's, the land mask is binary, and
the nl-oQM was fitted for ERA5-driven predictions. Outside the nl training box, the model runs zero-shot (warning).

Edit the settings below, then run from code/:  uv run python scripts/predict_gfs.py
"""

from __future__ import annotations

import concurrent.futures
import logging
import urllib.request

import eccodes
import numpy as np
import pandas as pd
import torch
import xarray as xr

from otprof import RUNS_ROOT
from otprof.pipelines.shared import P_LEVELS_ERA5, Z_AGL_WRF
from otprof.predict import make_pl_inputs, predict_profiles, warn_outside_nl

## Settings
LOCATIONS = {  # name: (lat, lon) in degrees north / east; the nearest GFS grid point is used
    "cabauw": (51.97, 4.93),
    "north_sea": (53.25, 4.25),
}
CYCLE = pd.Timestamp("2026-10-20 00:00")  # forecast initialisation (UTC), cycles at 00, 06, 12, 18
LEAD_HOURS = [0, 3, 6, 9, 12]  # hourly up to 120 h, then 3-hourly up to 384 h
Z_AGL = Z_AGL_WRF  # output heights [m above ground], increasing; default: mean heights of the 99 WRF training levels
APPLY_OQM = True  # add lcn2_oqm (nl-oQM) and the r0, lsi2 computed from it
MODEL_DIR = RUNS_ROOT / "models" / "wrf_pl"
OUT_FILE = RUNS_ROOT / "preds" / "gfs" / f"sqfh_gfs_{CYCLE:%Y%m%d%H}.nc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GFS_AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LEVELS = [int(p) for p in P_LEVELS_ERA5 / 100] + [20]  # hPa, bottom to top; 20 hPa only for the top gradient
LEVELS_PGRB2B = [875, 825, 775, 225, 175, 125]
# (GFS file, idx variable, idx level) -> name in make_pl_inputs. Pressure levels are added below.
GFS_SFC_FIELDS = {
    ("pgrb2", "PRES", "surface"): "p_sfc",
    ("pgrb2", "HGT", "surface"): "z_sfc",
    ("pgrb2", "LAND", "surface"): "lsm",
    ("pgrb2", "TMP", "2 m above ground"): "tk2",
    ("pgrb2", "UGRD", "10 m above ground"): "u10",
    ("pgrb2", "VGRD", "10 m above ground"): "v10",
    ("pgrb2", "PRMSL", "mean sea level"): "slp",
    ("sflux", "SHTFL", "surface"): "hfx",
    ("sflux", "LHTFL", "surface"): "lh",
    ("sflux", "FRICV", "surface"): "ust",
    ("sflux", "HPBL", "surface"): "blh",
}
GFS_PL_VARS = {"HGT": "z", "TMP": "tk", "UGRD": "u", "VGRD": "v"}


def gfs_url(cycle: pd.Timestamp, lead: int, file: str) -> str:
    """URL of a GFS file on AWS (the cycle directories have an atmos/ subdirectory since 2021-03-22)."""
    sub = "/atmos" if cycle >= pd.Timestamp("2021-03-22") else ""
    name = {
        "pgrb2": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}",
        "pgrb2b": f"gfs.t{cycle:%H}z.pgrb2b.0p25.f{lead:03d}",
        "sflux": f"gfs.t{cycle:%H}z.sfluxgrbf{lead:03d}.grib2",
    }[file]
    return f"{GFS_AWS}/gfs.{cycle:%Y%m%d}/{cycle:%H}{sub}/{name}"


def load_gfs_lead(cycle: pd.Timestamp, lead: int, locations: dict[str, tuple[float, float]]) -> xr.Dataset:
    """The fields of one lead time at the grid points nearest to `locations`, named as in make_pl_inputs."""
    pl_file = {p: "pgrb2b" if p in LEVELS_PGRB2B else "pgrb2" for p in LEVELS}
    keys = list(GFS_SFC_FIELDS) + [(pl_file[p], var, f"{p} mb") for var in GFS_PL_VARS for p in LEVELS]
    fcst = "anl" if lead == 0 else f"{lead} hour fcst"  # instantaneous fields only, not the interval averages

    # Byte ranges of the messages from the .idx inventories ("n:offset:d=...:VAR:LEVEL:FCST:")
    ranges = {}
    for file in {f for f, _, _ in keys}:
        url = gfs_url(cycle, lead, file)
        lines = [ln.split(":") for ln in urllib.request.urlopen(f"{url}.idx").read().decode().splitlines()]
        offsets = [int(ln[1]) for ln in lines] + [None]
        for i, ln in enumerate(lines):
            if (file, ln[3], ln[4]) in keys and ln[5] == fcst:
                ranges[(file, ln[3], ln[4])] = (url, offsets[i], offsets[i + 1])
    missing = set(keys) - set(ranges)
    if missing:
        raise ValueError(f"GFS fields not found for lead {lead} h: {sorted(missing)}")

    def _fetch(key):
        url, start, end = ranges[key]
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{'' if end is None else end - 1}"})
        gid = eccodes.codes_new_from_message(urllib.request.urlopen(req).read())
        nearest = [eccodes.codes_grib_find_nearest(gid, lat, lon % 360)[0] for lat, lon in locations.values()]
        eccodes.codes_release(gid)
        return key, nearest

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        nearest = dict(pool.map(_fetch, keys))

    ds = xr.Dataset({name: ("loc", [n["value"] for n in nearest[key]]) for key, name in GFS_SFC_FIELDS.items()})
    for var, name in GFS_PL_VARS.items():
        prof = [[n["value"] for n in nearest[(pl_file[p], var, f"{p} mb")]] for p in LEVELS]
        ds[name] = (("level", "loc"), np.array(prof))
    grid = nearest[("pgrb2", "PRES", "surface")]  # 0.25 deg grid point
    return ds.astype(np.float32).assign_coords(
        level=LEVELS,
        loc=list(locations),
        lat=("loc", [n["lat"] for n in grid]),
        lon=("loc", [((n["lon"] + 180) % 360) - 180 for n in grid]),
        time=cycle + pd.Timedelta(hours=lead),
        lead=lead,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warn_outside_nl(LOCATIONS)

    print(f"Downloading GFS {CYCLE:%Y-%m-%d %H} UTC, leads {LEAD_HOURS} h...")
    ds_gfs = xr.concat([load_gfs_lead(CYCLE, lead, LOCATIONS) for lead in LEAD_HOURS], dim="time")
    ds_gfs["slp"] = ds_gfs["slp"] / 100  # hPa
    ds_gfs["z_sfc"] = ds_gfs["z_sfc"].mean("time")  # static (orography)
    ds_gfs["ust"] = ds_gfs["ust"].clip(min=0.01)  # guard against 0 from GRIB packing (W71 CT2 must stay finite)
    ds_gfs = ds_gfs.transpose("time", "level", "loc")

    ds_X = make_pl_inputs(ds_gfs)
    z_agl = xr.DataArray(np.asarray(Z_AGL, dtype=np.float32), dims="level")
    ds_pred = predict_profiles(ds_X, z_agl, MODEL_DIR, apply_oqm=APPLY_OQM, device=DEVICE)
    ds_pred = ds_pred.assign_coords(lead=ds_gfs["lead"], lat=ds_gfs["lat"], lon=ds_gfs["lon"])
    ds_pred.attrs["gfs_cycle"] = f"{CYCLE:%Y-%m-%d %H:%M} UTC"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_pred.to_netcdf(OUT_FILE)
    print(ds_pred)
    print(f"Predictions written to {OUT_FILE}")
