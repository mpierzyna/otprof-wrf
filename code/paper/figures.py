"""Figures and score tables of the paper from the predictions in runs/preds/ (see analysis.py).

Writes paper/plots/<set>/ for the figure sets in PLOT_SETS (lcn2 profiles, structure functions, score plots
and LaTeX tables, KDE and qq panels of lcn2, r0 and lsi2), paper/plots/input_agreement/ (agreement of the ERA5
PL and WRF PL inputs) and the vertical levels figure. Also prints the fraction of r0 below 50 and 100 cm.

Run from code/:  uv run python paper/figures.py
"""

import logging
import pathlib
from typing import Tuple, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import analysis as ma
import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
from research_tools.ot import ot
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

sns.set_palette("colorblind")
plt.rcParams.update(
    {
        # "font.family": "serif",
        "font.size": 8,
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.dpi": 300,
        # "figure.labelsize": 8,
        "lines.linewidth": 0.75,
        "hatch.linewidth": 0.5,
    }
)

# BASELINE_UPPER = "true/pred_wrf_native"
# BASELINE_LOWER = "true/pred_hv_wrfpl"

# Scale figures because otherwise very small
FIG_WIDTH = 7.25  # in
FIG_WIDTH *= 0.9

# Fast KDE / qq panels (plot_hist_qq): binned KDE and qq from QQ_N_QUANTILES quantiles instead of the
# exact KDE (scipy, as seaborn) and every sorted value (pooled lcn2 has ~10^7 values per case). Set False
# to go back to the exact versions (much slower) if the figures look off.
FAST_PLOTTING = True
QQ_N_QUANTILES = 2000
# Sets with several runs (ensemble members, e.g. the canaries fits A, B, C) in the KDE, qq and structure
# function panels: "band" = median line and min-max band, "lines" = one thin line per run
ENSEMBLE_STYLE = "band"
# qq panel ticks per variable, the same on both axes: (major, minor) step in the variable's units, major
# ticks labelled. Variables not listed keep matplotlib's automatic major ticks with automatic minor ticks.
QQ_TICKS = {
    "lcn2": (5, 1),  # dex
    "r0": (50, 10),  # cm
    "lsi2": (1, 0.2),  # dex
}

# every n-th nl test sample; 10 while developing, 1 for the final figures
NL_SUBSAMPLE = 1

VARS_PRETTY = {
    "lcn2": r"$\log_{10}C_n^2$",
    "r0": r"$r_0$",
    "lsi2": r"$\log_{10}\sigma_I^2$",
    "z": r"$z$",  # height above ground
    "zt": r"$z_t$",  # terrain height
    # model input features (plot_input_agreement)
    "z_agl": r"$z$",  # height above ground (not renamed when loaded directly)
    "u": r"$u(z)$",
    "v": r"$v(z)$",
    "th": r"$\theta(z)$",
    "S": r"$S(z)$",  # vertical wind shear
    "dth_dz": r"$\Gamma(z)$",
    "ust": r"$u_*$",
    "hfx": r"$Q_H$",  # sensible heat flux
    "lh": r"$Q_E$",  # latent heat flux
    "blh": r"$h$",  # boundary layer height
    "u10": r"$u_{10}$",
    "v10": r"$v_{10}$",
    "tk2": r"$T_2$",
    "lct2_w71f": r"$[\log_{10}C_T^2]_s$",
}
# Model input features compared between WRF PL and ERA5 PL (plot_input_agreement, analysis.open_inputs): profiles on the common pressure
# levels and surface features; not the static, time and pressure features (identical in both)
INPUT_PROFILE = [
    # "z_agl",
    "u",
    "v",
    "th",
    "S",
    "dth_dz",
]
INPUT_SURFACE = [
    "ust",
    "hfx",
    "lh",
    "blh",
    # "u10",
    # "v10",
    # "tk2",
    # "slp",
    "lct2_w71f",
]


# Input features that modulate the turbulence (shaded in plot_input_agreement): gradients, surface fluxes and
# boundary layer, which produce or, depending on the sign (stable stratification, downward heat flux), destroy
# it; the others describe the large-scale state
# TURB_MODULATORS = ["S", "dth_dz", "ust", "hfx", "lh", "blh", "u10", "v10", "lct2_w71f"]
TURB_MODULATORS = []

VARS_UNITS = {
    "r0": "cm",
    "z": "m",
    "zt": "m",
}

# Case labels and styles. Canaries sets relabel the nl-trained model as zero-shot (CN_PRETTY).
EXP_PRETTY = {
    ma.DatasetKey.TRUE: "Ref.",
    ma.DatasetKey.WRF_NATIVE: "Ctl.",
    ma.DatasetKey.WRF_PL: "WRF-PL",
    ma.DatasetKey.ERA5_PL: "ERA5-PL no oQM",
    ma.DatasetKey.ERA5_PL_OQM_NL: "ERA5-PL oQM",
    ma.DatasetKey.ERA5_PL_OQM_CN: "ERA5-PL cn-oQM",
    ma.DatasetKey.ERA5_PL_DIRECT: "ERA5-PL direct",
    ma.DatasetKey.HV_WRFPL: "HV+W71",
}
CN_PRETTY = {
    ma.DatasetKey.ERA5_PL: "no oQM",
    ma.DatasetKey.ERA5_PL_OQM_NL: "NL-oQM",
    ma.DatasetKey.ERA5_PL_OQM_CN: "CI-oQM",
}

C_BLACK = "k"
C_BLUE = "C0"
C_ORANGE = "C1"
C_GREEN = "C2"
C_RED = "C3"
C_PURPLE = "C4"
C_BROWN = "C5"
C_PINK = "C6"
C_GREY = "C7"
C_YELLOW = "C8"
C_LIGHT_BLUE = "C9"

EXP_COLORS = {
    ma.DatasetKey.TRUE: C_BLACK,
    ma.DatasetKey.WRF_NATIVE: C_RED,  # red
    ma.DatasetKey.WRF_PL: C_PINK,  # pink
    ma.DatasetKey.ERA5_PL: C_BLUE,  # blue
    ma.DatasetKey.ERA5_PL_OQM_NL: C_GREEN,  # green
    ma.DatasetKey.ERA5_PL_OQM_CN: C_ORANGE,  # orange
    ma.DatasetKey.ERA5_PL_DIRECT: C_LIGHT_BLUE,  # light blue
    ma.DatasetKey.HV_WRFPL: C_GREY,  # grey
}

# One axis per row: dimensional, dimensionless skill, distribution shape. Metrics a variable does not have
# are left out (r_vert: profiles only). No R^2: with r, bias and cRMSE it adds nothing
# (R^2 = 2 r a - a^2 - (bias / std)^2, a = std ratio).
METRICS_ROWS = [["bias", "crmse"], ["r", "r_vert"], ["d_skew", "d_kurt"]]
METRICS_PRETTY = {
    "bias": r"Bias ($\to 0$)",
    "crmse": r"cRMSE ($\downarrow$)",
    "r": r"$r_\mathrm{temp}$ ($\uparrow$)",
    "r_vert": r"$r_\mathrm{vert}$ ($\uparrow$)",
    "d_skew": r"$\Delta$skew ($\to 0$)",
    "d_kurt": r"$\Delta$kurt ($\to 0$)",
}

PAIR_COLORS = {f"true/{e}": c for (e, c) in EXP_COLORS.items() if not e == ma.DatasetKey.TRUE}
PAIR_MARKERS = {
    f"true/{ma.DatasetKey.WRF_NATIVE}": "^",
    f"true/{ma.DatasetKey.WRF_PL}": "o",
    f"true/{ma.DatasetKey.ERA5_PL}": "o",
    f"true/{ma.DatasetKey.ERA5_PL_OQM_NL}": "o",
    f"true/{ma.DatasetKey.ERA5_PL_OQM_CN}": "s",
    f"true/{ma.DatasetKey.ERA5_PL_DIRECT}": "o",
    f"true/{ma.DatasetKey.HV_WRFPL}": "v",
}
PAIR_PRETTY = {f"true/{k}": v for k, v in EXP_PRETTY.items()}


def derived_vars(ds: ma.Dataset) -> dict[str, ma.Array]:
    """lcn2 of every case in `ds` and the r0 [cm] and lsi2 computed from it."""
    lcn2 = ds["lcn2"]
    r0 = lcn2.map(lambda x: ot.fried_r0_xr(cn2=10**x, z=ds.z, dim="bottom_top") * 100).rename("r0")
    lsi2 = lcn2.map(lambda x: np.log10(ot.scint_index_xr(cn2=10**x, z=ds.z, dim="bottom_top", mode="plane"))).rename(
        "lsi2"
    )
    return {"lcn2": lcn2, "r0": r0, "lsi2": lsi2}


def plot_scores(a: ma.Array, pairs_plot, a_runs: List[ma.Array] = ()) -> Tuple[plt.Figure, str]:
    """Score plot and LaTeX table. `a_runs` are further runs (seeds) of the same models: markers then
    show the mean over all runs with a min-max bar, and the table mean +- std."""
    metrics_runs = []
    for a_ in [a, *a_runs]:
        m_ = ma.compute_metrics_t(a_, dim=["sample", "bottom_top"] if a_.is_profile else ["sample"]).to_dataframe()
        metrics_rows = [r for r in ([m for m in row if m in m_.columns] for row in METRICS_ROWS) if r]
        metrics_runs.append(m_[sum(metrics_rows, [])].loc[pairs_plot])
    runs = pd.concat(metrics_runs, keys=range(len(metrics_runs)))
    metrics = runs.groupby(level=1, sort=False).mean().loc[pairs_plot]
    metrics_min = runs.groupby(level=1, sort=False).min().loc[pairs_plot]
    metrics_max = runs.groupby(level=1, sort=False).max().loc[pairs_plot]
    metrics_std = runs.groupby(level=1, sort=False).std().loc[pairs_plot]

    # Legend columns: all entries in one row if the labels are short, else rows of 2 or 3
    legend_rows = 2
    ncols = 2
    # A panel with a single metric (r_temp only, for r0 and lsi2) gets 1.5 rows of height, else it is cramped
    height_ratios = [max(len(r), 1.5) for r in metrics_rows]
    fig, axarr = plt.subplots(
        # height for a one-row legend, scaling with the number of metric rows (5 originally: 2 + 2 + 1),
        # plus one legend line ((1 + labelspacing) * fontsize) per further legend row
        figsize=(
            FIG_WIDTH * 0.5,
            FIG_WIDTH / 3.4 * sum(height_ratios) / 5.5
            + (legend_rows - 1) * (1 + plt.rcParams["legend.labelspacing"]) * plt.rcParams["font.size"] / 72,
        ),
        constrained_layout=True,
        nrows=len(metrics_rows),
        height_ratios=height_ratios,
    )

    for ax, metrics_subset in zip(axarr, metrics_rows):
        m_space = np.arange(len(metrics_subset))[::-1]
        for i, m in zip(m_space, metrics_subset):
            for p in pairs_plot:
                label = PAIR_PRETTY[p] if i == m_space[0] else None  # only label once
                ax.scatter(
                    metrics[m].loc[p],
                    i,
                    marker=PAIR_MARKERS[p],
                    label=label,
                    edgecolors=PAIR_COLORS[p],
                    facecolors=PAIR_COLORS[p],
                )
                if a_runs:  # spread over runs
                    ax.hlines(i, metrics_min[m].loc[p], metrics_max[m].loc[p], color=PAIR_COLORS[p], lw=1.5)

        ax.set_yticks(m_space)
        ax.set_ylim(-0.5, len(metrics_subset) - 0.5)
        ax.set_yticklabels([f"{METRICS_PRETTY[m]}" for m in metrics_subset])

        if set(metrics_subset) <= {"r", "r_vert"}:
            m_min = metrics[metrics_subset].min().min()
            # at least [0.5, 1]; from 0 if a value is lower (markers off the axis edge), below 0 only if needed
            ax.set_xlim(0.5 if m_min >= 0.5 else 0 if m_min >= 0.02 else m_min - 0.05, 1)
        else:  # bias, cRMSE, Δskew, Δkurt: zero is perfect
            ax.axvline(0, color="k", lw=0.5)

    # above the plots, centred on the figure: x in figure, y in axes coordinates (a figure legend
    # gets extra padding from constrained layout and shrinks the axes)
    # axarr[0].legend(ncols=ncols, bbox_to_anchor=(1, 1.1), loc="lower right", borderaxespad=0)
    axarr[0].legend(
        ncols=ncols,
        bbox_to_anchor=(0.5, 1),
        bbox_transform=mtransforms.blended_transform_factory(fig.transFigure, axarr[0].transAxes),
        loc="lower center",
        borderaxespad=1.2,  # gap to the axes in font sizes, independent of the axes height
    )

    # Convert dataframe to latex table (mean +- std over runs where they differ)
    metrics_latex = metrics.map(lambda v: f"{v:.3f}")
    if a_runs:
        has_spread = metrics_std > 1e-9
        metrics_latex[has_spread] = metrics_latex[has_spread] + metrics_std[has_spread].map(
            lambda v: rf" $\pm$ {v:.3f}"
        )
    metrics_latex = metrics_latex.T
    # two-line header cells, broken at the first space (e.g. ERA5-PL / no oQM); needs \usepackage{makecell}
    metrics_latex.columns = [
        rf"\makecell{{{PAIR_PRETTY[c].replace(' ', r'\\', 1)}}}" if " " in PAIR_PRETTY[c] else PAIR_PRETTY[c]
        for c in metrics_latex.columns
    ]
    metrics_latex = metrics_latex.reset_index()
    metrics_latex["index"] = metrics_latex["index"].map(METRICS_PRETTY)
    metrics_latex = metrics_latex.rename(columns={"index": ""})
    metrics_latex = metrics_latex.to_latex(
        escape=False,
        index=False,
        column_format="l" + "c" * (len(metrics_latex.columns) - 1),
    )

    return fig, metrics_latex


def kde(x: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE of `x` (Scott bandwidth, as seaborn) on a grid over [lo, hi], zero where there are no
    data. FAST_PLOTTING: binned (histogram on 4096 bins smoothed with the kernel), else exact (scipy, 512
    points)."""
    x = x[np.isfinite(x)]
    bw = np.std(x, ddof=1) * x.size ** (-1 / 5)
    if FAST_PLOTTING:
        counts, edges = np.histogram(x, bins=4096, range=(lo, hi))
        dx = edges[1] - edges[0]
        return (edges[:-1] + edges[1:]) / 2, gaussian_filter1d(counts.astype(float), sigma=bw / dx, mode="constant") / (
            x.size * dx
        )
    grid = np.linspace(lo, hi, 512)
    return grid, gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))(grid)


def plot_runs(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, color: str, label: str, lw: float) -> None:
    """Draw one case: `xs`, `ys` with shape (runs, points). One run: a line of width `lw`; several runs
    (ENSEMBLE_STYLE): median line and min-max band over a common x (the median of xs), or thin lines."""
    if len(ys) == 1:
        ax.plot(xs[0], ys[0], color=color, label=label, lw=lw)
    elif ENSEMBLE_STYLE == "band":
        x = np.median(xs, axis=0)
        ax.plot(x, np.median(ys, axis=0), color=color, label=label, lw=lw * 0.75)
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.25, lw=0)
    else:
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.plot(x, y, color=color, label=label if i == 0 else None, lw=lw * 0.5)


def plot_hist_qq(
    a: ma.Array,
    exps_plot,
    pairs_plot,
    xmin: float | None = None,
    xmax: float | None = None,
    ymax: float | None = None,
    a_runs: List[ma.Array] = (),
) -> plt.Figure:
    """KDE and qq plot of variable `a`. `a_runs` are further runs (ensemble members) of the same cases,
    drawn as set by ENSEMBLE_STYLE. In band mode the qq band is the spread of the per-run deviation
    (estimated - reference quantile) around the median reference quantile, so every run stays paired with
    its own reference. FAST_PLOTTING: binned KDE, qq from QQ_N_QUANTILES quantiles dense at both tails."""
    runs = [a, *a_runs]
    vmin = np.nanmin(a[ma.DatasetKey.TRUE].values) if xmin is None else xmin
    vmax = np.nanmax(a[ma.DatasetKey.TRUE].values) if xmax is None else xmax

    fig, (ax_hist, ax_qq) = plt.subplots(
        ncols=2,
        figsize=(FIG_WIDTH * 0.9, FIG_WIDTH / 3),
        constrained_layout=True,
        width_ratios=[2, 1.25],
    )
    unit = "" if not a.name in VARS_UNITS else f", {VARS_UNITS[a.name]}"

    # KDE on one grid for all cases and runs, covering the axis and every value plus 3 bandwidths
    lo, hi = vmin, vmax
    for a_ in runs:
        for e in exps_plot:
            x = a_[e].values.ravel()
            x = x[np.isfinite(x)]
            bw = np.std(x, ddof=1) * x.size ** (-1 / 5)
            lo, hi = min(lo, x.min() - 3 * bw), max(hi, x.max() + 3 * bw)
    for e in exps_plot:
        curves = [kde(a_[e].values.ravel(), lo, hi) for a_ in runs]
        plot_runs(
            ax_hist,
            np.array([c[0] for c in curves]),
            np.array([c[1] for c in curves]),
            EXP_COLORS[e],
            EXP_PRETTY[e],
            lw=1.5,
        )
    ax_hist.set_xlabel(VARS_PRETTY[a.name] + unit)
    ax_hist.legend()
    ax_hist.set_xlim(vmin, vmax)
    ax_hist.set_ylim(0, ymax)

    # qq
    q = (1 - np.cos(np.pi * np.linspace(0, 1, QQ_N_QUANTILES))) / 2  # probabilities, dense at both tails
    use_quantiles = FAST_PLOTTING or (a_runs and ENSEMBLE_STYLE == "band")  # bands need common probabilities
    for p in pairs_plot:
        e = p.split("/", 1)[1]
        if use_quantiles:
            xs = np.array([np.nanquantile(a_[ma.DatasetKey.TRUE].values.ravel(), q) for a_ in runs])
            ys = np.array([np.nanquantile(a_[e].values.ravel(), q) for a_ in runs])
        else:  # every sorted value (runs can differ in size: draw them one by one)
            xs = [np.sort(a_[ma.DatasetKey.TRUE].values.ravel()) for a_ in runs]
            ys = [np.sort(a_[e].values.ravel()) for a_ in runs]
        if a_runs and ENSEMBLE_STYLE == "band":
            x_med = np.median(xs, axis=0)
            plot_runs(ax_qq, np.tile(x_med, (len(runs), 1)), x_med + (ys - xs), PAIR_COLORS[p], PAIR_PRETTY[p], lw=1.5)
        elif a_runs or use_quantiles:
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax_qq.plot(
                    x,
                    y,
                    label=PAIR_PRETTY[p] if i == 0 else None,
                    lw=0.75 if a_runs else 1.5,
                    color=PAIR_COLORS[p],
                    rasterized=True,
                )
        else:
            ax_qq.scatter(xs[0], ys[0], label=PAIR_PRETTY[p], s=3, color=PAIR_COLORS[p], rasterized=True)

    ax_qq.plot([vmin, vmax], [vmin, vmax], ls="--", color="k")
    ax_qq.text(0.01, 0.99, "overestimation", transform=ax_qq.transAxes, va="top", ha="left", color="grey")
    ax_qq.text(0.99, 0.01, "underestimation", transform=ax_qq.transAxes, va="bottom", ha="right", color="grey")
    ax_qq.set_xlabel(f"Ref. {VARS_PRETTY[a.name]}" + unit)
    ax_qq.set_ylabel(f"Estimated {VARS_PRETTY[a.name]}" + unit)
    ax_qq.set_xlim(vmin, vmax)
    ax_qq.set_ylim(vmin, vmax)
    ax_qq.set_aspect("equal")
    for axis in (ax_qq.xaxis, ax_qq.yaxis):
        if a.name in QQ_TICKS:
            major, minor = QQ_TICKS[a.name]
            axis.set_major_locator(plt.MultipleLocator(major))
            axis.set_minor_locator(plt.MultipleLocator(minor))
        else:
            axis.set_minor_locator(mticker.AutoMinorLocator())

    return fig


def plot_sf(a: ma.Array, exps_plot, a_runs: List[ma.Array] = ()) -> plt.Figure:
    """Plot structure function of variable a. `a_runs` are further runs (ensemble members) of the same
    cases, drawn as set by ENSEMBLE_STYLE (all runs share the lag bins)."""
    dz_edges = ma.get_tf_edges(x_min=50, x_max=20_000, n_bins=100, tf=np.sqrt, tf_inv=lambda x: x**2)

    fig, (ax_full, ax_zoom) = plt.subplots(ncols=2, figsize=(FIG_WIDTH, FIG_WIDTH / 3), constrained_layout=True)
    for e in exps_plot:
        print(f"Computing structure function for {e}")
        sfs = [
            ma.compute_ensemble_sf(
                x=jnp.array(a_.z.isel(sample=slice(None, None, 10)).values),
                y=jnp.array(a_[e].isel(sample=slice(None, None, 10)).values),
                bin_edges=jnp.array(dz_edges),
            )
            for a_ in [a, *a_runs]
        ]
        is_valid = np.all([np.asarray(l_freq) >= 1e-3 for _, _, l_freq in sfs], axis=0)  # lag bins valid in every run
        xs = np.array([np.asarray(l)[is_valid] for l, _, _ in sfs])
        ys = np.array([np.asarray(s2)[is_valid] for _, s2, _ in sfs])
        for ax in [ax_full, ax_zoom]:
            plot_runs(ax, xs, ys, EXP_COLORS[e], EXP_PRETTY[e], lw=1)

    for ax in [ax_full, ax_zoom]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta z$, m")
        ax.margins(x=0)

    # only full axis
    ax_full.set_ylabel(rf"$S^2(\Delta z)$ of {VARS_PRETTY[a.name]}")
    if len(exps_plot) <= 5:
        ax_full.legend(ncols=2, loc="lower right")
    else:  # long legends would cover the curves
        fig.set_figheight(FIG_WIDTH / 2.4)
        fig.legend(*ax_full.get_legend_handles_labels(), ncols=4, loc="outside lower center")

    # Zoomed axis limits
    ax_zoom.set_xlim(2e3, None)
    ax_zoom.set_ylim(8e-1, 10)

    # Draw matching box on full axis
    ax_full.indicate_inset_zoom(ax_zoom, edgecolor="black")

    return fig


def plot_random_profiles(a: ma.Array, exps_plot, z_lr: xr.DataArray, ncols: int, nrows: int, seed: int) -> plt.Figure:
    """Plot random profiles from the triplet."""
    rng = np.random.default_rng(seed)
    inds = rng.choice(a.data[ma.DatasetKey.TRUE].sizes["sample"], ncols * nrows, replace=False)
    a_dict = a.as_dict()

    fig, axarr = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_WIDTH * nrows / 3),
        constrained_layout=True,
        sharex="all",
        sharey="all",
    )
    if axarr.ndim == 1:
        axarr = axarr[:, None]  # make 2d for consistent indexing

    legend_handles = []
    for ax, i in zip(axarr.flatten(), inds):
        # Plot profiles
        for e in exps_plot:
            v = a_dict[e]
            p = ax.plot(
                v.isel(sample=i),
                a.z.isel(sample=i),
                label=EXP_PRETTY[e],
                color=EXP_COLORS[e],
                zorder=100,
                lw=1.5,
            )
            if i == inds[0]:
                legend_handles.append(p[0])

        # Add meta data (use last v because all same loc and time)
        lat = v.isel(sample=i).lat.item()
        lat = f"{lat:.2f} °N" if lat >= 0 else f"{-lat:.2f} °S"
        lon = v.isel(sample=i).lon.item()
        lon = f"{lon:.2f} °E" if lon >= 0 else f"{-lon:.2f} °W"
        time = v.isel(sample=i).time.dt.strftime("%Y-%m-%d %H:%M").item()
        zt = f"{VARS_PRETTY['zt']} = {a_dict[ma.DatasetKey.TRUE].isel(sample=i).zt.item():.0f} {VARS_UNITS['zt']}"
        surface = "land" if a_dict[ma.DatasetKey.TRUE].isel(sample=i).lsm.item() > 0.5 else "sea"  # WRF land-sea mask

        ax.text(
            0.99,
            0.99,
            f"{lat}, {lon}\n{time}\n{zt}\n{surface}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=6.5,
        )

        # Add ERA5 levels in the background
        # This is not so helpful for the story
        # for z in z_lr:
        #     ax.axhline(z, color="lightgrey", lw=0.5, ls="--", zorder=-100)

        # Set sqrt scaling for y axis to stretch lower levels
        ax.set_yscale("function", functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)), lambda x: np.sign(x) * x**2))  # signed: no NaN below 0 (autoscaling), ticks there stay off the axis
        ax.margins(y=0)
        ax.set_ylim(0, None)

        # add major and minor ticks in y direction
        ax.yaxis.set_major_locator(plt.MultipleLocator(5000))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))

    for ax in axarr[:, 0]:
        ax.set_ylabel(f"{VARS_PRETTY['z']}, {VARS_UNITS['z']}")
    for ax in axarr[-1, :]:
        ax.set_xlabel(f"{VARS_PRETTY[a.name]}")

    # Legend above figure
    fig.legend(
        handles=legend_handles,
        fontsize=8,
        ncol=len(exps_plot) if len(exps_plot) <= 5 else 4,
        loc="outside upper center",
    )

    return fig


def plot_input_agreement(x_wrfpl: xr.Dataset, x_era5pl: xr.Dataset, a: ma.Array, exp: ma.DatasetKey) -> plt.Figure:
    """Agreement of the ERA5 PL inputs with the WRF PL inputs the model was trained on: r_temp and r_vert per
    input feature (compute_metrics, WRF PL as reference), turbulence modulators shaded and at the bottom. Lines:
    r_temp and r_vert of `a` (lcn2) predicted from ERA5 PL (`exp`) against the reference, the skill reached
    despite the input mismatch."""
    m_in = {
        f: ma.compute_metrics(
            ma.Pair(f, x_wrfpl[f], x_era5pl[f]),
            dim=["sample", "bottom_top"] if "bottom_top" in x_wrfpl[f].dims else ["sample"],
            per_level=True,
        )
        for f in x_wrfpl.data_vars
    }
    m_out = ma.compute_metrics(ma.Pair(a.name, a[ma.DatasetKey.TRUE], a[exp]), dim=["sample", "bottom_top"], per_level=True)

    # bottom to top: modulators, then the large-scale state, each sorted by r_temp
    modulators = sorted([f for f in m_in if f in TURB_MODULATORS], key=lambda f: float(m_in[f]["r"]))
    state = sorted([f for f in m_in if f not in TURB_MODULATORS], key=lambda f: float(m_in[f]["r"]))
    order = modulators + state
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 3, FIG_WIDTH * 0.45), constrained_layout=True)
    ax.axhspan(-0.5, len(modulators) - 0.5, color="0.92", lw=0, zorder=-10)
    # vertical along the right edge: no modulator agrees close to r = 1
    if modulators:
        ax.text(
            0.985, (len(modulators) - 1) / 2, "turbulence modulators", transform=ax.get_yaxis_transform(), rotation=90,
            ha="right", va="center", fontsize=7, color="0.4",
        )  # fmt: skip
    ax.scatter([float(m_in[f]["r"]) for f in order], y, marker="o", color=C_BLACK, label=f"Inp. {METRICS_PRETTY['r'].split(' (')[0]}", zorder=3)
    prof = [i for i, f in enumerate(order) if "r_vert" in m_in[f]]
    ax.scatter(
        [float(m_in[order[i]]["r_vert"]) for i in prof], prof, marker="D", facecolors="none", edgecolors=C_BLACK,
        label=f"Inp. {METRICS_PRETTY['r_vert'].split(' (')[0]}", zorder=3,
    )  # fmt: skip
    for m, ls in [("r", "--"), ("r_vert", ":")]:
        ax.axvline(
            float(m_out[m]), color=EXP_COLORS[exp], ls=ls, lw=1,
            label=f"{VARS_PRETTY[a.name]} {METRICS_PRETTY[m].split(' (')[0]}",
        )  # fmt: skip

    ax.set_yticks(y, [VARS_PRETTY[f] for f in order])
    ax.set_ylim(-0.5, len(order) - 0.5)
    r_min = min(float(m[k]) for m in m_in.values() for k in ["r", "r_vert"] if k in m)
    ax.set_xlim(min(0, r_min - 0.05), 1)
    ax.set_xlabel("Correlation")
    # above the axes, centred on the figure (as in plot_scores)
    ax.legend(
        loc="lower center", ncols=2, borderaxespad=0, bbox_to_anchor=(0.5, 1.02),
        bbox_transform=mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes),
        fontsize=7, handlelength=1.5, columnspacing=0.8, handletextpad=0.4,
    )  # fmt: skip
    return fig


def plot_markers() -> plt.Figure:
    """Simple figure to show marker styles used in other plots."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 10 * len(PAIR_COLORS), FIG_WIDTH / 10), constrained_layout=True)
    for i, p in enumerate(PAIR_COLORS):
        ax.scatter(i, 0, c=PAIR_COLORS[p], marker=PAIR_MARKERS[p])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_levels(data_root: pathlib.Path) -> plt.Figure:
    # Open native high-res X
    ds_hr = xr.open_dataset(data_root / "wrfnative_X_test.nc")
    ds_lr = xr.open_dataset(data_root / "wrfpl_X_test.nc")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 3, FIG_WIDTH / 1.5), constrained_layout=True)

    z = ds_hr["z_agl"].mean("sample")
    p = ds_hr["p"].mean("sample") / 100
    ax.scatter(p + 50, z, s=2, label="WRF +50hPa")

    z = ds_lr["z_agl"].mean("sample")
    p = ds_lr["p"].mean("sample") / 100
    ax.scatter(p, z, s=2, marker="D", label="ERA5-PL")

    ax.set_yscale("function", functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)), lambda x: np.sign(x) * x**2))  # signed: no NaN below 0 (autoscaling), ticks there stay off the axis
    ax.set_xlabel("Pressure, hPa")
    ax.set_ylabel("Height, m")
    ax.legend()

    # add major and minor ticks in y direction
    ax.yaxis.set_major_locator(plt.MultipleLocator(5000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))
    ax.set_xlim(0, None)
    # ax.set_yticklabels([0, 1000, 5000, 10000, 15000, 20000, 25000])

    ax.margins(0)

    return fig


def print_percentiles(a: ma.Array, a_vals: List[float], exp_plots) -> None:
    """Compute percentiles for values in `a_vals` and print them."""
    res = {}
    for e in exp_plots:
        q = np.linspace(0, 1, 200)  # quantiles to compute
        q_v = a[e].quantile(q).values  # compute quantiles
        res[e] = np.interp(a_vals, q_v, q)  # interpolate
    res = pd.DataFrame(res, index=a_vals, columns=exp_plots)
    print(res)


def save_and_show(fig: plt.Figure, fpath: pathlib.Path, **kwargs) -> None:
    fig.savefig(fpath, transparent=True, pad_inches=0, **kwargs)
    fig.show()


if __name__ == "__main__":
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Figure sets: <name> -> (dataset, further runs of the same cases, cases)
    #   main                          nl 2017 test: in-domain ladder with the final framework (decoupled
    #                                 training + bulk lcn2 output QM fitted on nl)
    #   appendix                      nl 2017 test: decoupled with and without nl oQM vs trained
    #                                 directly on ERA5 PL
    #   canaries_<land|all>_fit<X>    canaries, zero-shot nl model on the 40 days not in group X (A, B, C):
    #                                 without oQM, with the nl oQM, with the canaries oQM fitted on X
    #                                 (land and sea maps); <land>: WRF land points only, <all>: incl. sea
    #   canaries_<land|all>           the three fits together: every fit drawn as a thin line per case
    #                                 (Ref. and HV too, each fit is scored on other days), scores as mean
    #                                 and min-max; example profiles of fit A
    # The nl sets take ~1 h at full resolution (KDEs over all nl profiles and levels), the canaries sets
    # a few minutes each.
    PLOT_SETS = [
        "main", "appendix",
        "canaries_land", "canaries_land_fitA", "canaries_land_fitB", "canaries_land_fitC",
        "canaries_all", "canaries_all_fitA", "canaries_all_fitB", "canaries_all_fitC",
    ]  # fmt: skip

    exp_main = [
        ma.DatasetKey.TRUE,
        ma.DatasetKey.WRF_NATIVE,
        ma.DatasetKey.WRF_PL,
        ma.DatasetKey.ERA5_PL_OQM_NL,
        ma.DatasetKey.HV_WRFPL,
    ]
    exp_app = [
        ma.DatasetKey.TRUE,
        ma.DatasetKey.ERA5_PL_OQM_NL,
        ma.DatasetKey.ERA5_PL,
        ma.DatasetKey.ERA5_PL_DIRECT,
    ]
    exp_cn = [
        ma.DatasetKey.TRUE,
        ma.DatasetKey.ERA5_PL,
        ma.DatasetKey.ERA5_PL_OQM_NL,
        ma.DatasetKey.ERA5_PL_OQM_CN,
        ma.DatasetKey.HV_WRFPL,
    ]

    logger.info("Opening datasets...")
    sets = {}
    if {"main", "appendix"} & set(PLOT_SETS):
        ds_nl = ma.open_datasets(ma.NL_PRED_DIR, use_dask=False, files=ma.NL_FILES)
        ds_nl = ds_nl.isel({"sample": slice(None, None, NL_SUBSAMPLE)})
        sets["main"] = (ds_nl, [], exp_main)
        sets["appendix"] = (ds_nl, [], exp_app)
    for subset in ["land", "all"]:
        if any(name.startswith(f"canaries_{subset}") for name in PLOT_SETS):
            ds_fits = {fit: ma.open_canaries(fit, subset) for fit in ["A", "B", "C"]}
            for fit, d in ds_fits.items():
                sets[f"canaries_{subset}_fit{fit}"] = (d, [], exp_cn)
            sets[f"canaries_{subset}"] = (ds_fits["A"], [ds_fits["B"], ds_fits["C"]], exp_cn)

    EXP_PRETTY_DEFAULT = dict(EXP_PRETTY)
    for out_dir in PLOT_SETS:
        ds, ds_runs, exps = sets[out_dir]
        labels = {**EXP_PRETTY_DEFAULT, **(CN_PRETTY if out_dir.startswith("canaries") else {})}
        EXP_PRETTY.update(labels)
        PAIR_PRETTY.update({f"true/{k}": v for k, v in labels.items()})
        (plot_dir / out_dir).mkdir(parents=True, exist_ok=True)

        vars_main = derived_vars(ds)
        vars_runs = [derived_vars(d) for d in ds_runs]
        pairs = [f"true/{e}" for e in exps if e != ma.DatasetKey.TRUE]
        print_percentiles(vars_main["r0"], [50, 100], exps)
        save_and_show(
            plot_random_profiles(vars_main["lcn2"], exps_plot=exps, z_lr=ds.z_lr, ncols=4, nrows=3, seed=42),
            plot_dir / out_dir / "lcn2_profiles.pdf",
        )
        lcn2_runs = [v["lcn2"] for v in vars_runs]
        save_and_show(plot_sf(vars_main["lcn2"], exps_plot=exps, a_runs=lcn2_runs), plot_dir / out_dir / "lcn2_sf.pdf")

        # canaries: cap the r0 density axis, the narrow HV+W71 peak would flatten the other cases
        r0_ymax = 0.03 if out_dir.startswith("canaries") else None
        hist_limits = {
            "lcn2": {},
            "r0": {"xmin": 0, "xmax": 150, "ymax": r0_ymax},
            "lsi2": {"xmin": -2.6, "xmax": -0.5},
        }
        for var, limits in hist_limits.items():
            a_runs = [v[var] for v in vars_runs]
            fig, metrics_latex = plot_scores(vars_main[var], pairs_plot=pairs, a_runs=a_runs)
            save_and_show(fig, plot_dir / out_dir / f"{var}_scores.pdf")
            (plot_dir / out_dir / f"{var}_scores.tex").write_text(metrics_latex)
            save_and_show(
                plot_hist_qq(vars_main[var], exps_plot=exps, pairs_plot=pairs, a_runs=a_runs, **limits),
                plot_dir / out_dir / f"{var}_hist_qq.pdf",
                dpi=650,
            )
        plt.close("all")

    # Input agreement ERA5 PL vs WRF PL in plots/input_agreement,
    # per domain: nl on the same samples as the main set, canaries on all 60 land days (all groups: no
    # canaries oQM, which is fitted on one)
    input_sets = {}
    if "main" in PLOT_SETS:
        x_nl = ma.open_inputs([ma.NL_PRED_DIR], land=False, profile=INPUT_PROFILE, surface=INPUT_SURFACE)
        input_sets["nl"] = (*(x.isel({"sample": slice(None, None, NL_SUBSAMPLE)}) for x in x_nl), ds_nl)
    if "canaries_land" in PLOT_SETS:
        x_cn = ma.open_inputs(
            [root / ma.CN_TRUTH_DIR for root in ma.CN_GROUP_ROOTS.values()], land=True, profile=INPUT_PROFILE, surface=INPUT_SURFACE
        )
        input_sets["canaries_land"] = (*x_cn, ma.open_canaries(None, "land"))
    (plot_dir / "input_agreement").mkdir(parents=True, exist_ok=True)
    for domain, (x_wrfpl, x_era5pl, ds) in input_sets.items():
        labels = {**EXP_PRETTY_DEFAULT, **(CN_PRETTY if domain.startswith("canaries") else {})}
        EXP_PRETTY.update(labels)
        vars_ = derived_vars(ds)
        exp = ma.DatasetKey.ERA5_PL_OQM_NL  # the framework (nl oQM in both domains)
        save_and_show(
            plot_input_agreement(x_wrfpl, x_era5pl, vars_["lcn2"], exp=exp),
            plot_dir / "input_agreement" / f"input_agreement_{domain}.pdf",
        )
        plt.close("all")

    if "main" in PLOT_SETS:
        save_and_show(plot_levels(ma.NL_PRED_DIR), plot_dir / "levels.pdf")
        save_and_show(plot_markers(), plot_dir / "marker_legend.pdf")
