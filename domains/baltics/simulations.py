"""Simulations of this workspace. Create their directories with `wrf-massive init-sims simulations.py <variable>`."""

import datetime
from wrf_massive.base import Simulation
from wrf_massive.sim_gen import get_spaced_sims

# Settings are inserted into namelist.tmpl.input by matching names, e.g.
# `physics__bl_pbl_physics` fills `{{ physics__bl_pbl_physics }}`.
mynn25 = {
    # MYNN 2.5 without EDMF (mixing length 1)
    "physics__sf_sfclay_physics": "5",
    "physics__bl_pbl_physics": "5",
    "physics__bl_mynn_closure": "2.5",
    "physics__bl_mynn_mixlength": "1",
    "physics__bl_mynn_edmf": "0",
    "physics__bl_mynn_edmf_mom": "0",
    "physics__bl_mynn_edmf_tke": "0",
    "physics__sf_urban_physics": "0",  # no urban model
}


def get_1y_sims(net_sim_days: int = 5, step: int = 4, offset: int = 0) -> list[Simulation]:
    global_begin = datetime.datetime(2017, 1, 1)
    global_end = datetime.datetime(2021, 1, 1)

    return get_spaced_sims(
        global_begin,
        global_end,
        settings=mynn25,
        net_sim_days=net_sim_days,
        step=step,
        offset=offset,
    )

# nl: offset=0 (indirectly from old sims); alps: offset=1; baltics: offset=2; pyrenees: offset=3; 
sims_1y = get_1y_sims(offset=2)  
sims_1y_half = sims_1y[::2]

# Test simulation with 1 day of usable data. The 12 h warmup is added before `begin` automatically.
sim_test = Simulation(
    begin="2020-01-01T00:00:00",
    end="2020-01-02T00:00:00",
    warmup_h=12,
    sim_dir="test_1",
    settings=mynn25,
)
