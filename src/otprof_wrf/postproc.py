# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Post-processing of the OTProf WRF runs."""

from typing import List

from wrf_massive.stages.postproc import PostProcFn
from wrf_massive.stages.postproc.base import TVarList
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage, fn_ct2_cn2


class OTProfCn2PostProcStage(Cn2PostProcStage):
    """CT2/Cn2 post-processing with the variables stored for OTProf.

    Needs the WRF output of the OTProf `myoutfields.txt` (e.g. `domains/alps`): besides the Cn2 inputs the 2 m and
    surface fields, the MYNN variances QSQ/COV, RAINNC and the WRF-Solar cloud diagnostics (`solar_diagnostics = 1`).
    Pass `myoutfields_path` to have that checked when the pipeline is loaded.
    """

    extract_vars: TVarList = [
        # Vertical coordinates
        "z",
        "HGT",
        # Pressure
        "p",
        "slp",  # sea level pressure
        "PSFC",  # surface pressure
        # Wind
        "u_met",
        "v_met",
        "U10",
        "V10",
        "w",
        # Temperature
        "th",  # potential temperature
        "T2",
        "TSK",
        # Humidity
        "rh",
        "Q2",
        # Surface fluxes
        "LH",
        "HFX",
        "UST",
        # CT2 and Cn2
        "QKE",
        ("EL_PBL", "bottom_top_stag"),
        "TSQ",
        "QSQ",  # sigma_q^2; not used
        "COV",  # covariance sigma_qT; not used
        # Diagnostics
        "PBLH",
        "LANDMASK",
        # Precip and clouds
        "RAINNC",
        "CLDFRAC2D",
        "CBASEHT_TOT",
        "CTOPHT_TOT",
        "LWP_TOT",
        "IWP_TOT",
        "WP_TOT_SUM",
    ]

    postproc_fns: List[PostProcFn] = [
        fn_ct2_cn2,
    ]
