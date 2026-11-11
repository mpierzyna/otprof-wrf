"""Simulations of this workspace. Create their directories with `wrf-massive init-sims simulations.py <variable>`."""

from wrf_massive.base import  Simulation

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

# Test simulation with 1 day of usable data. The 12 h warmup is added before `begin` automatically.
sim_test = Simulation(
    begin="2020-01-01T00:00:00",
    end="2020-01-02T00:00:00",
    warmup_h=12,
    sim_dir="test_1",
    settings=mynn25,
)
