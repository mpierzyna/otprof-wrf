# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipeline for a single 5-day WRF run over the Canary Islands.

Forcing comes from the TU Delft CERRA mirror. WPS pre-processing runs on DelftBlue (`p_preproc`) and WRF +
post-processing on Snellius (`p_snellius`); the active pipeline is selected via `env.yaml`.
"""

import datetime
import pathlib

from wrf_massive.base import Pipeline, Resources, Simulation, Stage
from wrf_massive.stages.forcing import PullCerraStage
from wrf_massive.stages.misc import GarbageCollectStage, MarkDone
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.stages.wps import WPSStage
from wrf_massive.stages.wrf import WRFStage
from wrf_massive.workspace import load_env


def _update_resources(stage: Stage, **resources) -> Stage:
    """Helper to update n_tasks of a stage. Returns deep copy."""
    import copy

    stage = copy.deepcopy(stage)
    stage.resources = stage.resources.model_copy(update=resources)
    return stage


# Load host-specific environment settings
env = load_env()  # env file selected by `wrf-massive --env-file` (default: env.yaml)


# Preproc
_cerra = PullCerraStage(
    work_dir="1_forcing",
    remote_path="tudelft:staff-umbrella/HBaki/CERRA",
    remote_flist_path="CERRA_files.txt.gz",
    n_transfers=4,
    resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"),
)

_wps = WPSStage(
    work_dir="2_wps",
    forcing_dir=_cerra.work_dir,  # 1_forcing
    namelist_tmpl_path="namelist.tmpl.wps",
    **env["wps"],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="16G"),  # serial WPS
)

# Forcing data no longer needed after WPS finished -> clear space.
_forcing_gc = GarbageCollectStage(
    work_dir=_cerra.work_dir,  # 1_forcing
    glob_pattern="**/*.grb",  # CDS: flat, mirror: in year subdirs
    armed=True,
    run_cond_fn=_wps.is_done,  # double-check WPS completion
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

_wrf = WRFStage(
    work_dir="3_wrf",
    met_em_dir=_wps.work_dir,  # 2_wps
    namelist_tmpl_path="namelist.tmpl.input",
    myoutfields_path="myoutfields.txt",
    **env["wrf"],
    resources=Resources(n_tasks=48, cpus_per_task=1, mem_per_cpu="1G"),  # chaos: 128 cores available
)

_cn2 = Cn2PostProcStage(
    work_dir="4_postproc",
    wrfout_dir=_wrf.work_dir,  # 3_wrf
    domain=1,
    compression=True,
    run_parallel=True,
    resources=Resources(n_tasks=1, cpus_per_task=8, mem_per_cpu="1GB"),
)


def cn2_files_exist(s: Simulation) -> bool:
    """Check if any CN2 output files exist for simulation s."""
    cn2_dir = _cn2.get_work_dir(s)
    return len(list(pathlib.Path(cn2_dir).glob("wrfout*cn2.nc"))) > 0


# met_em*.nc no longer needed once Cn2 output exists -> clear space.
_wps_gc = GarbageCollectStage(
    work_dir=_wps.work_dir,  # 2_wps
    glob_pattern="met_em*.nc",
    armed=True,
    run_cond_fn=cn2_files_exist,
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

# Mark whole simulation dir as done when all stages complete.
_sim_done = MarkDone(
    work_dir=".",
    run_cond_fn=cn2_files_exist,
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)


if env["machine"] == "delftblue":
    # WPS pre-processing with forcing from the TU Delft CERRA mirror. WPS runs dmpar on 4 tasks.
    # compute_p1: 48 cores, 185GB RAM, 150GB SSD, mem-per-cpu=3996M
    # compute_p2: 64 cores, 256GB RAM, 360GB SSD, mem-per-cpu=4000M
    # check https://doc.dhpc.tudelft.nl/delftblue/DHPC-hardware/ for details.
    p_preproc = Pipeline(
        cerra=_update_resources(
            _cerra,
            n_tasks=1,
            cpus_per_task=8,
        ),
        wps=_update_resources(
            _wps,
            n_tasks=12,  # 1/4 of phase 1 node
            cpus_per_task=1,
            mem_per_cpu="1000M",
            walltime=datetime.timedelta(hours=1),
        ),
    )

if env["machine"] == "snellius":
    # For Snellius: run WRF and postproc with more ressources.
    # check https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions
    # tmp-dir behaviour is configured per substage. Copies, so the shared `_wrf`/`_cn2` instances keep their defaults.
    TMP_ROOT = pathlib.Path("/scratch-shared/mpierzyna/")
    _wrf_tmp = _wrf.model_copy(
        update={
            "tmp_work_root": TMP_ROOT,
            "tmp_teardown_globs": [  # move only settings and scripts back
                "setup_wrf.sh",
                "run_wrf.sh",
                "namelist.input",
                "myoutfields.txt",
                ".gitignore",
            ],
            "tmp_skip_teardown": True,  # leave symlink in place so Cn2 can postproc
        }
    )
    _cn2_tmp = _cn2.model_copy(
        update={
            "tmp_work_root": TMP_ROOT,
            "tmp_skip_teardown": True,
        }
    )

    p_snellius = Pipeline(
        # One full genoa node for WRF
        wrf=_update_resources(
            _wrf_tmp,
            n_tasks=192,
            cpus_per_task=1,
            mem_per_cpu="1750M",  # all the RAM
            walltime=datetime.timedelta(hours=12),
        ),
        cn2=_update_resources(
            _cn2_tmp,
            n_tasks=1,
            cpus_per_task=48,
            mem_per_cpu="4000M",
            walltime=datetime.timedelta(minutes=30),
        ),
    )
