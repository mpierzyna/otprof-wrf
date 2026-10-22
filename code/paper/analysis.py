"""Data classes, loading and metrics for the paper figures (figures.py).

Predictions are read from runs/preds/ (scripts/predict_nl.py and scripts/predict_canaries.py):
  nl (2017 test)        : NL_PRED_DIR, NL_FILES
  canaries (60 days)    : CN_GROUP_ROOTS, CN_FILES; the 20-day month groups A (months 1, 4, 7, 10),
                          B (2, 5, 8, 11) and C (3, 6, 9, 12)
The truth (WRF native) and the HV+W71 reference come with every prediction dir (open_datasets).
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import warnings
from enum import StrEnum
from typing import Callable, Dict, List, Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from otprof import RUNS_ROOT
from scipy import stats

# Stay on CPU
jax.config.update("jax_platforms", "cpu")

# Type for dimension argument
TDim = Union[None, str, List[str]]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel("INFO")


class DatasetKey(StrEnum):
    TRUE = "true"  # WRF native
    WRF_NATIVE = "pred_wrf_native"  # nl model, WRF native input
    WRF_PL = "pred_wrfpl"  # nl model, WRF interpolated to ERA5 PL (training modality)
    ERA5_PL = "pred_era5pl"  # nl model, ERA5 PL input (decoupled)
    ERA5_PL_OQM_NL = "pred_era5pl_oqm_nl"  # ERA5_PL + bulk lcn2 output QM fitted on nl (nl training years)
    ERA5_PL_OQM_CN = "pred_era5pl_oqm_cn"  # ERA5_PL + bulk lcn2 output QM fitted on one canaries group
    ERA5_PL_DIRECT = "pred_era5pl_direct"  # model trained directly on nl ERA5 PL (no decoupling)
    HV_WRFPL = "pred_hv_wrfpl"  # Hufnagel-Valley + Walters (1971) surface value from WRF PL


# nl: prediction files relative to the pred dir
NL_PRED_DIR = RUNS_ROOT / "preds" / "nl"
NL_FILES = {
    DatasetKey.WRF_NATIVE: "wrfnative_y_pred.nc",
    DatasetKey.WRF_PL: "wrfpl_y_pred.nc",
    DatasetKey.ERA5_PL: "era5pl_y_pred.nc",
    DatasetKey.ERA5_PL_OQM_NL: "era5pl_oqm_y_pred.nc",
    DatasetKey.ERA5_PL_DIRECT: "era5pl_direct_y_pred.nc",
}

# canaries: pred dir per month group (truth and HV reference included), zero-shot nl model. Files per
# fit group of the canaries oQM: the map fitted on group <fit> exists only in the two other groups.
CN_GROUP_ROOTS = {g: RUNS_ROOT / "preds" / "canaries" / g for g in ["A", "B", "C"]}
CN_TRUTH_DIR = "."
CN_FILES = {
    fit: {
        DatasetKey.WRF_PL: "wrfpl_y_pred.nc",
        DatasetKey.ERA5_PL: "era5pl_y_pred.nc",
        DatasetKey.ERA5_PL_OQM_NL: "era5pl_oqmnl_y_pred.nc",
        DatasetKey.ERA5_PL_OQM_CN: f"era5pl_oqmfit{fit}_y_pred.nc",
    }
    for fit in CN_GROUP_ROOTS
}


@dataclasses.dataclass
class Pair:
    """Pair of one variable (DataArrays)"""

    name: str
    a: xr.DataArray
    b: xr.DataArray

    @property
    def is_profile(self) -> bool:
        return "bottom_top" in self.a.dims


@dataclasses.dataclass
class Array:
    """Array of one variable from different datasets"""

    name: str
    z: xr.DataArray
    data: Dict[DatasetKey, xr.DataArray]

    def __post_init__(self):
        if DatasetKey.TRUE not in self.data:
            raise ValueError(f"Missing key {DatasetKey.TRUE} in Array data")

    def map(self, fn: Callable, *args, **kwargs) -> Array:
        """Apply function `fn` to each DataArray in the triplet."""
        return Array(
            name=self.name,
            z=self.z,
            data={k: fn(v, *args, **kwargs) for k, v in self.data.items()},
        )

    def map_(self, fname: str, *args, **kwargs) -> Array:
        """Apply DataArray method `fname` to each DataArray in the triplet."""
        return Array(
            name=self.name,
            z=self.z,
            data={k: getattr(v, fname)(*args, **kwargs) for k, v in self.data.items()},
        )

    def __getitem__(self, item: str | DatasetKey) -> xr.DataArray:
        """Get DataArray for dataset key `item`."""
        return self.data[item]

    def as_dict(self) -> Dict[DatasetKey, xr.DataArray]:
        return self.data

    def as_pairs(self) -> Dict[str, Pair]:
        pairs = {}
        true_da = self.data[DatasetKey.TRUE]
        for k, v in self.data.items():
            if k != DatasetKey.TRUE:
                pairs[f"{DatasetKey.TRUE}/{k}"] = Pair(self.name, true_da, v)
        return pairs

    def sel(self, sel) -> Array:
        """Select data from each DataArray in the triplet."""
        return self.map_("sel", sel)

    def isel(self, indexers) -> Array:
        """Index data from each DataArray in the triplet."""
        return self.map_("isel", indexers)

    def rename(self, name: str) -> Array:
        """Rename the triplet variable."""
        return Array(name=name, z=self.z, data=self.data)

    @property
    def is_profile(self) -> bool:
        return "bottom_top" in self.data[DatasetKey.TRUE].dims


@dataclasses.dataclass
class Dataset:
    """Triplet of datasets"""

    z: xr.DataArray  # vertical coordinate for all datasets
    z_lr: xr.DataArray  # average vertical coordinate for ERA5 dataset
    data: Dict[DatasetKey, xr.Dataset]

    def __post_init__(self):
        if DatasetKey.TRUE not in self.data:
            raise ValueError(f"Missing key {DatasetKey.TRUE} in Dataset data")

    def __getitem__(self, v) -> Array:
        """Get triplet for variable `v`."""
        return Array(name=v, z=self.z, data={k: ds[v] for k, ds in self.data.items()})

    def sel(self, sel) -> Dataset:
        """Select data from each Dataset in the triplet."""
        return Dataset(
            z=self.z.sel(sel),
            z_lr=self.z_lr,
            data={k: ds.sel(sel) for k, ds in self.data.items()},
        )

    def isel(self, indexers) -> Dataset:
        """Index data from each Dataset in the triplet."""
        return Dataset(
            z=self.z.isel(indexers),
            z_lr=self.z_lr,
            data={k: ds.isel(indexers) for k, ds in self.data.items()},
        )


def open_datasets(
    pred_dir: pathlib.Path | str,
    use_dask: bool,
    files: Dict[DatasetKey, str],
    truth_dir: str = ".",
) -> Dataset:
    """Open the predictions `files` (relative to `pred_dir`), the truth and inputs of the WRF PL test
    split in `pred_dir / truth_dir`, and the HV reference `pred_dir / wrfpl_ref.nc`."""
    pred_dir = pathlib.Path(pred_dir)
    chunks = {"sample": "auto", "bottom_top": -1} if use_dask else None

    # Get ERA5 average vertical height
    X_wrfpl = xr.open_dataset(pred_dir / truth_dir / "wrfpl_X_test.nc", chunks=chunks).set_index(sample=["time", "loc"])
    z_era5 = X_wrfpl["z_agl"].mean("sample")  # average vertical coordinate over all samples

    # True WRF NATIVE, with terrain height and WRF land-sea mask per sample (profile plot labels)
    y_true = xr.open_dataset(pred_dir / truth_dir / "wrfpl_y_test.nc", chunks=chunks).set_index(sample=["time", "loc"])
    if not X_wrfpl.indexes["sample"].equals(y_true.indexes["sample"]):
        raise ValueError("Samples of the inputs do not match the truth")
    y_true = y_true.assign_coords(
        zt=("sample", (y_true["z"] - y_true["z_agl"]).isel(bottom_top=0).values),
        lsm=("sample", X_wrfpl["lsm"].values),
    )

    # Literature references based on WRF PL inputs
    y_wrfpl_ref = xr.open_dataset(pred_dir / "wrfpl_ref.nc", chunks=chunks).set_index(sample=["time", "loc"])

    data = {DatasetKey.TRUE: y_true}
    for k, f in files.items():
        data[k] = (
            xr.open_dataset(pred_dir / f, chunks=chunks).set_index(sample=["time", "loc"]).rename(level="bottom_top")
        )
    data[DatasetKey.HV_WRFPL] = y_wrfpl_ref[["lcn2_hv"]].rename({"lcn2_hv": "lcn2"}).assign_coords(y_true.coords)

    # Plots compare raw values, so all datasets must hold the same samples in the same order
    for k, ds in data.items():
        if not ds.indexes["sample"].equals(y_true.indexes["sample"]):
            raise ValueError(f"Samples of {k} do not match the truth")
    return Dataset(z=y_true["z_agl"], z_lr=z_era5, data=data)


def open_canaries(fit: Literal["A", "B", "C"] | None, subset: Literal["land", "all"]) -> Dataset:
    """Canaries evaluation data for the canaries oQM fitted on group `fit`: the two other month groups
    (40 days, out of sample for every case), concatenated; `subset` "land" keeps the WRF land points.
    `fit` None: all three groups (60 days), without the canaries oQM."""
    files = CN_FILES[fit] if fit else {k: f for k, f in CN_FILES["A"].items() if k != DatasetKey.ERA5_PL_OQM_CN}
    ds_groups = []
    for g, root in CN_GROUP_ROOTS.items():
        if g == fit:
            continue
        d = open_datasets(root, use_dask=False, files=files, truth_dir=CN_TRUTH_DIR)
        if subset == "land":
            is_land = xr.open_dataset(root / CN_TRUTH_DIR / "wrfpl_X_test.nc")["lsm"].values > 0.5  # WRF mask
            d = d.isel({"sample": np.flatnonzero(is_land)})
        ds_groups.append(d)
    return Dataset(
        z=xr.concat([d.z for d in ds_groups], "sample"),
        z_lr=sum(d.z_lr for d in ds_groups) / len(ds_groups),
        data={k: xr.concat([d.data[k] for d in ds_groups], "sample") for k in ds_groups[0].data},
    )


def open_inputs(
    dirs: List[pathlib.Path], land: bool, profile: List[str], surface: List[str]
) -> Tuple[xr.Dataset, xr.Dataset]:
    """WRF PL and ERA5 PL test inputs in `dirs`, concatenated, with the (time, loc) sample index: the
    `profile` features on the common ERA5 pressure levels, NaN where underground in either, and the
    `surface` features. `land`: WRF land points only."""
    xs_wrfpl, xs_era5pl = [], []
    for d in dirs:
        x_wrfpl = xr.open_dataset(d / "wrfpl_X_test.nc").rename(bottom_top_h="bottom_top")
        x_era5pl = xr.open_dataset(d / "era5pl_X_test.nc")
        if not all(np.array_equal(x_wrfpl[c].values, x_era5pl[c].values) for c in ["time", "loc"]):
            raise ValueError(f"Samples of the WRF PL and ERA5 PL inputs in {d} do not match")
        valid = (x_wrfpl["valid"] > 0.5) & (x_era5pl["valid"] > 0.5)
        keep = np.flatnonzero(x_wrfpl["lsm"].values > 0.5) if land else slice(None)  # WRF land-sea mask
        for x, xs in [(x_wrfpl, xs_wrfpl), (x_era5pl, xs_era5pl)]:
            xs.append(xr.merge([x[profile].where(valid), x[surface]], compat="override").isel(sample=keep))
    return tuple(xr.concat(xs, "sample").set_index(sample=["time", "loc"]) for xs in (xs_wrfpl, xs_era5pl))


def compute_metrics(pair: Pair, dim: TDim, per_level: bool) -> xr.Dataset:
    """Compute metrics between a pair of DataArrays with a (time, loc) multi-index `sample`.
    r and cRMSE measure the temporal placement: computed per location along time, then averaged over
    the locations, so the differences between locations (site climatology) do not count. If
    `per_level` is True and the DataArrays are profiles, all metrics but d_skew, d_kurt and r_vert are
    computed per level and then averaged over the levels; along the vertical r would be dominated by
    the mean decrease of Cn2 with height (the climatological mean profile alone reaches r ~0.9 there).
    """

    logger.debug(f"Computing metrics for {pair.name} over dimensions: {dim}")

    def _pearson_r(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute Pearson correlation coefficient between two DataArrays. All-NaN slices (e.g. a level
        underground at a location throughout, or (time, loc) combinations without a sample) give NaN."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # std of the all-NaN slices
            a_mean = a.mean(dim=dim)
            b_mean = b.mean(dim=dim)
            cov = ((a - a_mean) * (b - b_mean)).mean(dim=dim)
            a_std = a.std(dim=dim)
            b_std = b.std(dim=dim)
        return cov / (a_std * b_std)

    def _wasserstein(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute 1D Wasserstein distance between two DataArrays."""
        from scipy.stats import wasserstein_distance

        logger.debug("Computing Wasserstein distance for profiles")

        return xr.apply_ufunc(
            wasserstein_distance,
            a,
            b,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def _bias(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute bias between two DataArrays."""
        return b.mean(dim) - a.mean(dim)

    def _rmse(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute RMSE between two DataArrays."""
        return np.sqrt(((a - b) ** 2).mean(dim=dim))

    def _crmse(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute centered RMSE between two DataArrays."""
        a_ = a - a.mean(dim)
        b_ = b - b.mean(dim)
        return _rmse(a_, b_, dim=dim)

    def _shape_diff(a: xr.DataArray, b: xr.DataArray) -> Tuple[float, float]:
        """Skewness and excess kurtosis of `b` minus those of the reference `a`, all values pooled.
        Differences, not ratios: both moments are dimensionless and near 0 for near-Gaussian
        distributions (skewness can change sign), where a ratio is unstable. 0 = same shape;
        d_skew > 0: tail on the high side too long, d_kurt > 0: tails too heavy / peak too sharp."""
        a, b = a.values.ravel(), b.values.ravel()
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        return float(stats.skew(b) - stats.skew(a)), float(stats.kurtosis(b) - stats.kurtosis(a))

    metrics = {
        "lat": pair.a["lat"][0],
        "lon": pair.a["lon"][0],
    }

    # Distribution shape: pooled over all values (samples and levels), as in the KDE and qq plots, not
    # per profile
    metrics["d_skew"], metrics["d_kurt"] = _shape_diff(pair.a, pair.b)

    def _r_vert(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        """Vertical structure: Pearson r along the vertical per profile, averaged over the profiles, of
        the anomalies from the own time-mean profile per location (reference and prediction each their
        own), so that neither the mean decrease of Cn2 with height nor a systematic error of the mean
        profile (that is the bias) counts. `a`, `b` on the unstacked (time, loc) grid: combinations
        without a sample are all-NaN profiles, their r is NaN and skipped in the mean."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # std of the all-NaN profiles
            return _pearson_r(a - a.mean("time"), b - b.mean("time"), dim="bottom_top").mean()

    # Temporal skill per location: on the unstacked (time, loc) grid, where combinations without a sample
    # are NaN and skipped
    a_u, b_u = pair.a.unstack("sample"), pair.b.unstack("sample")
    dim_t = "sample"
    dims = [dim] if isinstance(dim, str) else list(dim)

    if pair.is_profile and per_level:
        metrics["r_vert"] = _r_vert(a_u, b_u)

        # For profiles, we compute metrics first per level, then average over the levels.
        dim_ = [d for d in dims if d != dim_t]

        metrics["bias"] = _bias(pair.a, pair.b, dim=dim_t).mean(dim=dim_)
        metrics["rmse"] = _rmse(pair.a, pair.b, dim=dim_t).mean(dim=dim_)
        metrics["crmse"] = _crmse(a_u, b_u, dim="time").mean(dim=["loc", *dim_])
        metrics["r"] = _pearson_r(a_u, b_u, dim="time").mean(dim=["loc", *dim_])

        # Compute Wasserstein distance for all profiles
        # Not used
        # wd = _wasserstein(pair.a, pair.b, dim=dim_z)
        # metrics["wasserstein"] = wd.mean(dim=dim_)
    else:
        dim_u = ["time" if d == dim_t else d for d in dims]  # all of dim, but per location
        metrics["bias"] = _bias(pair.a, pair.b, dim=dim)
        metrics["rmse"] = _rmse(pair.a, pair.b, dim=dim)
        metrics["crmse"] = _crmse(a_u, b_u, dim=dim_u).mean(dim="loc")
        metrics["r"] = _pearson_r(a_u, b_u, dim=dim_u).mean(dim="loc")

    return xr.Dataset(metrics)


def compute_metrics_t(t: Array, dim: TDim, per_level: bool = True) -> xr.Dataset:
    """Compute metrics DataFrame for all pairs in a Triplet."""
    logger.debug(f"Computing metrics for triplet {t.name}")
    pairs = t.as_pairs()
    metrics_list = []
    for n, pair in pairs.items():
        logger.debug(f"Computing metrics for pair: {n}")
        metrics = compute_metrics(pair, dim=dim, per_level=per_level)
        metrics = metrics.expand_dims(pair=[n])
        metrics_list.append(metrics)
    return xr.concat(metrics_list, dim="pair")


@jax.jit
def compute_ensemble_sf(
    x: jnp.ndarray,
    y: jnp.ndarray,
    bin_edges: jnp.ndarray,
    p: int = 2,
    use_abs: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ensemble structure function S^p(l) = <[y(x+l) - y(x)]^p> for non-uniformly sampled data using JAX.

    Steps are
    1. For each realization, compute all pairwise distances and increments
    2. Bin the displacements into lag bins and accumulate sums and counts
    3. Average over all realizations

    Parameters
    ----------
    x: jnp.ndarray
        Shape (N, n) - Non-uniform sampling positions, unit: length
    y: jnp.ndarray
        Shape (N, n) - Signal values, unit: signal
    num_lags: int
        Number of bins for lag distance l
    p: int
        Order of the structure function (default is 2 for S2)
    """
    # Determine global scale for binning and setup bins
    # max_l = jnp.max(x) - jnp.min(x)  # unit: length
    # bin_edges = jnp.linspace(100, max_l, num_lags + 1)  # unit: length
    # bin_edges = jnp.logspace(2, jnp.log10(max_l), num_lags + 1)  # unit: length
    num_lags = len(bin_edges) - 1

    # Precompute upper triangular indices for pairwise differences
    n = x.shape[1]
    ii, jj = jnp.triu_indices(n, k=1)

    def scan_body(carry, inputs):
        sum_acc, count_acc = carry
        xk, yk = inputs

        # Compute pairwise distances
        dx_pairs = jnp.abs(xk[ii] - xk[jj])

        # Compute pairwise increments (abs if specified, order p)
        if use_abs:
            dy_p_pairs = jnp.abs(yk[ii] - yk[jj]) ** p  # unit: signal^p
        else:
            dy_p_pairs = (yk[ii] - yk[jj]) ** p  # unit: signal^p

        # Bin the pairwise distances and accumulate sums and counts
        bin_sums, _ = jnp.histogram(dx_pairs, bins=bin_edges, weights=dy_p_pairs)  # weights act like sum of y per bin
        bin_counts, _ = jnp.histogram(dx_pairs, bins=bin_edges)

        return (sum_acc + bin_sums, count_acc + bin_counts), None

    # Initialize accumulators
    init_carry = (jnp.zeros(num_lags), jnp.zeros(num_lags))

    # Efficiently loops over N realizations with fixed memory overhead
    (total_sums, total_counts), _ = jax.lax.scan(scan_body, init_carry, (x, y))

    Sp = jnp.where(total_counts > 0, total_sums / total_counts, 0.0)  # ensemble averaging, unit: signal^p
    dx = (bin_edges[:-1] + bin_edges[1:]) / 2  # unit: length
    dx_freq = total_counts / jnp.sum(total_counts)  # unit: fraction

    return dx, Sp, dx_freq


def get_tf_edges(x_min: float, x_max: float, n_bins: int, tf: Callable | None, tf_inv: Callable | None) -> np.ndarray:
    """Get bin edges in transformed space for uniform binning in original space."""
    edges_tf = np.linspace(tf(x_min), tf(x_max), n_bins)
    dx = edges_tf[1] - edges_tf[0]  # in tf'd space, uniform sampling
    edges_tf = edges_tf - 0.5 * dx  # shift to get left edges
    edges_tf = np.append(edges_tf, edges_tf[-1] + dx)  # add rightmost edge
    edges = tf_inv(edges_tf)
    return edges
