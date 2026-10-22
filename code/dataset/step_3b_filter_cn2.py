"""Exclude Cn2 profiles with NaN, too high r0, or too high scintillation index."""

# todo: filter also all astro clim params based on identified validity masks

import xarray as xr
import pathlib
from research_tools.reporter import BaseReport
import matplotlib.pyplot as plt


ROOT_FEATURES = pathlib.Path("3_features")


def plot_quant_hist(v: xr.DataArray) -> plt.Figure:
    fig, ax = plt.subplots()
    v.plot.hist(ax=ax, bins=100)
    ax.axvline(v.quantile(0.99), color="C0", linestyle="--", label="99%")
    ax.axvline(v.quantile(0.995), color="C1", linestyle="--", label="99.5%")
    ax.axvline(v.quantile(0.999), color="C2", linestyle="solid", label="99.9%")  # the one we use
    ax.axvline(v.quantile(1), color="C3", linestyle="--", label="100%")
    ax.legend()
    return fig


# Load dataset with all profiles
ds_wrf = xr.open_dataset(ROOT_FEATURES / "wrf_all.nc")

with BaseReport("Cn2 profile filtering", ROOT_FEATURES / "filtering.html") as r:
    r0 = ds_wrf["r0_full"]
    r0_th = r0.quantile(0.999).drop_vars("quantile")
    r0_valid = r0 < r0_th
    print(f"{(~r0_valid).sum().item()} profiles have r0 > {r0_th.item():.2e} m")
    r.add_mpl_fig(fig=plot_quant_hist(r0), caption="Histogram Fried parameter")

    lsi2 = ds_wrf["lsi2_full"]
    lsi2_th = lsi2.quantile(0.999).drop_vars("quantile")
    lsi2_valid = lsi2 < lsi2_th
    print(f"{(~lsi2_valid).sum().item()} profiles have scintillation index > {10**lsi2_th.item():.2f}")
    r.add_mpl_fig(fig=plot_quant_hist(lsi2), caption="Histogram log10 scintillation index")

    cn2 = ds_wrf["cn2"]
    cn2_invalid = (cn2 == 0).any("bottom_top")
    cn2_valid = ~cn2_invalid
    print(f"{(cn2_invalid).sum().item()} profiles have Cn2 == 0 at any level")

    # Discard invalid profiles. Use .where because it preserves dimensions.
    valid_profiles = r0_valid & lsi2_valid & cn2_valid
    n_valid_profiles = valid_profiles.sum().item()
    print(
        f"Keeping {n_valid_profiles} profiles out of {valid_profiles.size} "
        f"({100 * n_valid_profiles / valid_profiles.size:.1f}%)."
    )

    # Discard also astroclim parameters where r0 or lsi2 are invalid
    valid_ac = r0_valid & lsi2_valid
    n_valid_ac = valid_ac.sum().item()
    for v in ds_wrf.data_vars:
        if v.startswith(("r0_", "tau0_", "th0_", "lsi2_")):
            ds_wrf[v] = ds_wrf[v].where(valid_ac)
    print(f"Filtered astroclimate parameters ({n_valid_ac/valid_ac.size:.1%} of valid targets remain).")

    # Save
    ds_wrf["cn2"] = ds_wrf["cn2"].where(valid_profiles)
    ds_wrf.to_netcdf(ROOT_FEATURES / "wrf_filtered.nc", engine="netcdf4")
