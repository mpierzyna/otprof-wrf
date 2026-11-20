# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zarr store settings shared by all OTProf domains, so every domain encodes its variables alike.

Rules and benchmarks: `submodules/wrf-massive/src/wrf_massive/store/DESIGN.md` (Encoding). 2D and static fields
are not listed and stay lossless (they are ≈ 1 % of the data).
"""

import pathlib

from wrf_massive.store.zarr_store import BitRound, FixedScaleOffset, ZarrStore

# Time-invariant output: stored once per domain, not per frame
OTPROF_STATIC_VARS = ["HGT", "LANDMASK"]

# 3D fields of the union of all domains' variables (a domain without one of them ignores its entry)
OTPROF_ENCODING = {
    # Differenced/differentiated by loaders: absolute or near-absolute precision
    "th": FixedScaleOffset(offset=300.0, step=1e-3),  # 1 mK, int32: any θ from -2.1e6 to 2.1e6 K
    "z": BitRound(keepbits=16),  # ≤ 7.6e-6 relative (≤ 0.03 m at 3 km)
    "p": BitRound(keepbits=16),  # ≤ 0.8 Pa at 1000 hPa
    # Winds and humidity: ≤ 1.2e-4 relative
    "u_met": BitRound(keepbits=12),
    "v_met": BitRound(keepbits=12),
    "w": BitRound(keepbits=12),
    "rh": BitRound(keepbits=12),
    # Turbulence: ≤ 4.9e-4 relative, i.e. 2e-4 decades in log10 space
    "cn2": BitRound(keepbits=10),
    "ct2": BitRound(keepbits=10),
    "QKE": BitRound(keepbits=10),
    "TSQ": BitRound(keepbits=10),
    "QSQ": BitRound(keepbits=10),
    "COV": BitRound(keepbits=10),
    "EL_PBL": BitRound(keepbits=10),
}


def otprof_store(path: str | pathlib.Path, domain: str) -> ZarrStore:
    """The OTProf store's group of `domain` (all domains share `path`, see `store_path` in env_<machine>.yaml)."""
    return ZarrStore(path=path, domain=domain, static_vars=OTPROF_STATIC_VARS, encoding=OTPROF_ENCODING)
