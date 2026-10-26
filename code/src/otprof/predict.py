"""Inference: best checkpoint lookup, model predictions, the WRF PL -> ERA5 PL input swap of the decoupled
model, the Hufnagel-Valley reference profile, the output quantile mapping (oQM) of ERA5-driven lcn2
predictions, and predictions from pressure-level data of any source (ERA5, GFS) at given times and locations."""

from __future__ import annotations

import pathlib
import warnings

import joblib
import lightning as L
import numpy as np
import torch
import xarray as xr
import yaml
from research_tools.ot import most, ot, profiles

from otprof import pipelines
from otprof.datasets import ProfileDataset
from otprof.experiments import SqueezeFormerExp
from otprof.lit import LitSqueezeFormerH
from otprof.logging import get_logger
from otprof.pipelines.shared import P_LEVELS_ERA5, WRF_TF_CONFIG, _lz, add_X_time_features
from otprof.transform import QuantileMapper

logger = get_logger()


def get_best_ckpt(model_path: pathlib.Path) -> pathlib.Path:
    """Get best checkpoint from `model_path` based on validation score."""

    def _get_score(f: pathlib.Path) -> float:
        data = torch.load(f, map_location="cpu", weights_only=False)
        (ckpt_state,) = [v for k, v in data["callbacks"].items() if k.startswith("ModelCheckpoint")]
        return ckpt_state["current_score"].item()

    ckpt_files = sorted((model_path / "checkpoints").glob("*.ckpt"))
    ckpt_scores = [_get_score(f) for f in ckpt_files]
    best_ckpt = ckpt_files[torch.tensor(ckpt_scores).argmin()]
    logger.info(f"Using best checkpoint: {best_ckpt} with score {min(ckpt_scores):.4f}")
    return best_ckpt


def predict(exp: SqueezeFormerExp, model_path: pathlib.Path, out_dir: pathlib.Path, out_basename: str) -> None:
    """Predict the test split of `exp` with the best checkpoint in `model_path`. Writes the unscaled test
    inputs and truth (`<out_basename>_X_test.nc`, `_y_test.nc`) and the predictions (`_y_pred.nc`) to `out_dir`."""
    assert model_path.exists(), f"Model path {model_path} does not exist."
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = get_best_ckpt(model_path)
    loader_test = exp.get_torch_loader(stage="test")
    model = exp.get_lit_model(data_train=loader_test.dataset)  # only shapes and variable order are used

    # Save also the untransformed test data
    X_test, y_test = exp.dp.split("test", transform=False)
    X_test.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_X_test.nc")
    y_test.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_y_test.nc")

    trainer = L.Trainer(accelerator="auto", devices=1, num_nodes=1, enable_checkpointing=False, logger=False)
    preds = trainer.predict(model, loader_test, ckpt_path=best_ckpt)

    y_pred, y_conf = zip(*preds)  # predictions and confidence
    y_pred = torch.concat(y_pred, dim=0)  # (samples, levels, vars)

    # Inverse trafo
    ds_y_pred = loader_test.dataset.unstack_y(y_pred, sample_coord="self")
    _, ds_y_pred = exp.dp.inverse_transform(ds_y=ds_y_pred)
    ds_y_pred.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_y_pred.nc")
    logger.info(f"Predictions saved to {out_dir / out_basename}_*.nc")


def wrf_to_era5(exp: SqueezeFormerExp, scaler_dir: pathlib.Path | None = None) -> SqueezeFormerExp:
    """Copy of the WRF PL experiment `exp` fed with ERA5 PL inputs (same domain, split and targets). The
    input scaler is the WRF one of `exp`. `scaler_dir` (a saved `data_pipeline` dir) loads the trained
    scalers into the ERA5 pipeline before setup; required for out-of-domain inference. Without it (in
    domain), the ERA5 pipeline fits its own scalers on the train split and keeps its output scaler, which
    is fit on the same WRF targets."""
    features = exp.features + ["m"]  # m is needed for HV profile
    dp_era5 = pipelines.era5_wrf.get_era5pl_wrf_h(exp.dp.domain)
    if scaler_dir is not None:
        dp_era5.load_ood_scalers(scaler_dir)
    dp_era5 = pipelines.era5_wrf.era5_match_wrf(dp_era5=dp_era5.setup(), dp_wrf=exp.dp, features=features)
    dp_era5.sel_train, dp_era5.sel_val, dp_era5.sel_test = exp.dp.sel_train, exp.dp.sel_val, exp.dp.sel_test
    return exp.model_copy(update={"name": f"{exp.name}_era5", "dp": dp_era5}, deep=True)


def predict_ref(exp: SqueezeFormerExp, out_dir: pathlib.Path, out_basename: str) -> None:
    """Hufnagel-Valley profile with the Walters (1971) surface value (HV+W71) from the inputs of `exp`,
    evaluated at the target heights. Writes `<out_basename>_ref.nc` with `lcn2_hv`."""
    X_test, y_test = exp.dp.split("test", transform=False)
    # The WRF pipelines put inputs (and native targets) on bottom_top_h
    if "bottom_top_h" in X_test.dims:
        X_test = X_test.rename(bottom_top_h="bottom_top")
    if "bottom_top_h" in y_test.dims:
        y_test = y_test.rename(bottom_top_h="bottom_top")

    # Convert surface CT2 from W71 to Cn2
    p_0 = X_test["p"].isel(bottom_top=0) / 100  # hPa
    tk = X_test["tk2"]
    ct2_0 = 10 ** X_test["lct2_w71f"]
    cn2_0 = (79e-6 * p_0 / tk**2) ** 2 * ct2_0

    logger.info("Computing Hufnagel-Valley Cn2 profile")
    cn2_hv = profiles.hufnagel_valley_xr(
        cn2_0=cn2_0,
        m=X_test["m"],  # use X for W computation
        z=X_test["z_agl"],
        z_eval=y_test["z_agl"],  # evaluate at the target levels
        dim="bottom_top",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"lcn2_hv": np.log10(cn2_hv)}).reset_index("sample").to_netcdf(out_dir / f"{out_basename}_ref.nc")
    logger.info(f"Reference profiles saved to {out_dir / out_basename}_ref.nc")


def output_qm_lcn2(
    truth_fit: xr.Dataset,
    pred_fit: xr.Dataset,
    pred_test: xr.Dataset,
    groups_fit: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
    n_quantiles: int = 100,
    save_path: pathlib.Path | None = None,
) -> np.ndarray:
    """Bulk output quantile mapping of predicted lcn2 profiles; returns the corrected lcn2 of `pred_test`
    as (sample, level).

    One monotone transfer function for the whole column (values of all levels pooled), fit on `pred_fit`
    (model predictions, `level` dim) against `truth_fit` (reference, `bottom_top` dim) for the same samples
    and applied to `pred_test`. Sine-spaced quantiles without the end points resolve the tails, which
    dominate r0 and lsi2 (see QuantileMapper). `groups_*` (labels per sample, e.g. land / sea) fit and apply
    one mapping per group. `save_path` saves the fitted mappers as a joblib dict {group: QuantileMapper}.
    """
    groups_fit = np.zeros(truth_fit.sizes["sample"], int) if groups_fit is None else groups_fit
    groups_test = np.zeros(pred_test.sizes["sample"], int) if groups_test is None else groups_test
    truth_fit, pred_fit = truth_fit[["lcn2"]], pred_fit[["lcn2"]].rename(level="bottom_top")
    lcn2 = pred_test["lcn2"].transpose("sample", "level").values.copy()
    qms = {}
    for g in np.unique(groups_test):
        qm = QuantileMapper(n_quantiles=n_quantiles, vars=["lcn2"], per_level=False, grid="sine")
        qm.fit(X_obs=truth_fit.isel(sample=groups_fit == g), X_model=pred_fit.isel(sample=groups_fit == g))
        test = pred_test[["lcn2"]].isel(sample=groups_test == g).rename(level="bottom_top")
        lcn2[groups_test == g] = qm.transform(test)["lcn2"].transpose("sample", "bottom_top").values
        qms[g.item()] = qm
    if save_path is not None:
        joblib.dump(qms, save_path)
        logger.info(f"Output quantile mapping saved to {save_path}")
    return lcn2


## Predictions from pressure-level data at user times and locations (scripts/predict_era5.py, predict_gfs.py)

# ERA5 grid points of the nl training data (lat, lon range in degrees)
NL_TRAIN_BOX = {"lat": (50.25, 54.0), "lon": (2.75, 8.25)}
# The W71 surface CT2 clips the Obukhov length to its 2.5 / 97.5 percentiles over the dataset (dataset/step_3a); these
# are the bounds of the nl ERA5 data, so lct2_w71f is computed as for the training inputs.
W71_L_BOUNDS_NL = (-1003.6894, 1003.8965)  # m


def warn_outside_nl(locations: dict[str, tuple[float, float]]) -> None:
    """Warn for every location whose nearest 0.25 deg grid point lies outside the nl training box."""
    (lat_min, lat_max), (lon_min, lon_max) = NL_TRAIN_BOX["lat"], NL_TRAIN_BOX["lon"]
    for name, (lat, lon) in locations.items():
        lat_g, lon_g = round(lat * 4) / 4, round((((lon + 180) % 360) - 180) * 4) / 4
        if not (lat_min <= lat_g <= lat_max and lon_min <= lon_g <= lon_max):
            warnings.warn(
                f"Location '{name}' ({lat}, {lon}) is outside the nl training domain (lat {lat_min}..{lat_max}, "
                f"lon {lon_min}..{lon_max}). The model runs zero-shot there and the nl-oQM may not apply; "
                "expect larger errors, in particular over complex terrain.",
                stacklevel=2,
            )


def make_pl_inputs(ds: xr.Dataset) -> xr.Dataset:
    """SQF-H inputs (time, loc, bottom_top) on the 30 ERA5 pressure levels 1000..30 hPa, as dataset/step_3a and the
    ERA5 PL pipeline build them. Gradients live on the half level above each pressure level.

    `ds` holds, in the conventions of the training data (ERA5 after dataset/step_2a):
    - on dim `level` (the 31 levels 1000..20 hPa, bottom to top, coordinate in hPa): `z` (height above MSL [m]),
      `tk` [K], `u`, `v` [m/s];
    - surface fields (time, loc): `p_sfc` [Pa], `ust` [m/s], `hfx`, `lh` (upward positive [W/m2]), `blh` [m],
      `u10`, `v10` [m/s], `tk2` [K], `slp` [hPa], `lsm` [0..1]; terrain height `z_sfc` [m] (loc).
    """
    if not np.array_equal(ds["level"].values, list(P_LEVELS_ERA5 / 100) + [20]):
        raise ValueError(f"Expected the levels {list(P_LEVELS_ERA5 / 100) + [20]} hPa, got {ds['level'].values}.")
    p = (ds["level"] * 100).astype(int).drop_vars("level")  # Pa
    ds = ds.drop_vars("level")  # positional levels, bottom to top
    full = {"level": slice(0, len(P_LEVELS_ERA5))}  # 1000..30 hPa; 20 hPa only enters the top gradient
    z_agl = (ds["z"] - ds["z_sfc"]).isel(full)
    th = (ds["tk"] * (100000 / p) ** ((1.4 - 1) / 1.4)).astype(np.float32)

    # Vertical gradients between consecutive levels
    dz = ds["z"].diff("level")
    S = np.sqrt((ds["u"].diff("level") / dz) ** 2 + (ds["v"].diff("level") / dz) ** 2).astype(np.float32)
    dth_dz = (th.diff("level") / dz).astype(np.float32)

    # W71 surface CT2 at 2 m
    shfx = ds["hfx"] / 1.216e3  # W/m2 -> K m/s
    L = -(ds["ust"] ** 3) * ds["tk2"] / (0.4 * shfx * 9.81)
    ct2_w71f = most.w71_flux_ct2(T_st=-shfx / ds["ust"], z=2.0, L=L, L_bounds=W71_L_BOUNDS_NL).clip(1e-8, None)

    ds_X = xr.Dataset(
        {
            "z_agl": z_agl,
            "p": p.isel(full),
            "u": ds["u"].isel(full),
            "v": ds["v"].isel(full),
            "th": th.isel(full),
            "S": S,
            "dth_dz": dth_dz,
            **{v: ds[v] for v in ["ust", "hfx", "lh", "blh", "u10", "v10", "tk2", "slp", "lsm"]},
            "lct2_w71f": np.log10(ct2_w71f),
            "valid": (p.isel(full) < ds["p_sfc"]).astype("float32"),
            "lz": _lz(z_agl),
            "z_sfc": ds["z_sfc"],
            "lp_sfc": np.log(ds["p_sfc"]),
        }
    ).rename(level="bottom_top")
    X_time, _ = add_X_time_features(ds_X, None)
    return ds_X.assign(X_time)


def predict_profiles(
    ds_X: xr.Dataset,
    z_agl: xr.DataArray,
    model_dir: pathlib.Path,
    apply_oqm: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4096,
) -> xr.Dataset:
    """Predict with the best checkpoint in `model_dir` from the inputs `ds_X` (time, loc, bottom_top, see
    make_pl_inputs) at the heights `z_agl` [m above ground] (dim level, optionally also time and loc).

    Returns the targets on (time, loc, level) and r0 [cm] and lsi2 (1550 nm, plane wave, as in the paper) integrated
    over `z_agl`. With `apply_oqm`, also `lcn2_oqm` with the nl output quantile mapping (`model_dir/oqm_lcn2.joblib`)
    and the r0_oqm, lsi2_oqm computed from it.
    """
    exp_config = yaml.safe_load((model_dir / "exp_config.yml").read_text())
    scaler_X = joblib.load(model_dir / "data_pipeline" / "scaler_X.joblib")
    scaler_y = joblib.load(model_dir / "data_pipeline" / "scaler_y.joblib")
    lit = LitSqueezeFormerH.load_from_checkpoint(get_best_ckpt(model_dir), map_location="cpu")
    model = lit.model._orig_mod.to(device).eval()  # uncompiled module, same weights

    # Samples as in the training data: stacked time and location
    times, locs = ds_X["time"], ds_X["loc"]
    z_agl = z_agl.broadcast_like(ds_X["lsm"]).transpose("time", "loc", "level")
    ds_frc = xr.Dataset({"z_agl": z_agl, "lz": _lz(z_agl)}).stack(sample=("time", "loc")).transpose("sample", ...)
    data = ProfileDataset(
        ds_X=scaler_X.transform(ds_X.stack(sample=("time", "loc")).transpose("sample", ...)[exp_config["features"]]),
        vars_X=exp_config["features"],
        ds_y=scaler_y.transform(ds_frc),  # lz is not scaled
        vars_y=[],
        vars_y_frc=exp_config["forcings"],
    )
    assert data.vars_X.index("valid") == lit.hparams.idx_valid and data.vars_X.index("lz") == lit.hparams.idx_lz

    with torch.no_grad():
        y = [
            model(data.X[i : i + batch_size].to(device), data.y_frc[i : i + batch_size, :, 1].to(device))[0].cpu()
            for i in range(0, len(data), batch_size)
        ]
    y = torch.concat(y).numpy().reshape(times.size, locs.size, z_agl.sizes["level"], -1)

    # Unscale and invert the log transforms of the targets
    ds_y = xr.Dataset({v: (("time", "loc", "level"), y[..., i]) for i, v in enumerate(exp_config["targets"])})
    ds_y = scaler_y.inverse_transform(ds_y)
    for v, (_, fn_inv) in WRF_TF_CONFIG.items():
        if v in ds_y:
            ds_y[v] = fn_inv(ds_y[v])
    ds_y = ds_y.assign_coords(time=times, loc=locs)

    # nl output quantile mapping of lcn2 (one transfer function for all levels)
    if apply_oqm:
        qm = joblib.load(model_dir / "oqm_lcn2.joblib")[0]
        ds_y["lcn2_oqm"] = qm.transform(ds_y[["lcn2"]])["lcn2"].astype(np.float32)

    # Integrated parameters from the profiles
    z = z_agl.drop_vars(["time", "loc"], errors="ignore")
    for v in [v for v in ["lcn2", "lcn2_oqm"] if v in ds_y]:
        cn2, suffix = 10 ** ds_y[v], v.removeprefix("lcn2")
        ds_y[f"r0{suffix}"] = ot.fried_r0_xr(cn2=cn2, z=z, dim="level") * 100
        ds_y[f"lsi2{suffix}"] = np.log10(ot.scint_index_xr(cn2=cn2, z=z, dim="level", mode="plane"))

    units = {"lcn2": "log10(m^-2/3)", "QKE": "m2 s-2", "TSQ": "K2", "EL_PBL": "m", "r0": "cm", "lsi2": "1"}
    for v in ds_y:
        ds_y[v].attrs["units"] = units[v.removesuffix("_oqm")]
    return ds_y.assign_coords(
        z_agl=z_agl.assign_attrs(units="m", long_name="Height above ground"),
        z_sfc=ds_X["z_sfc"].assign_attrs(units="m", long_name="Terrain height"),
        lsm=ds_X["lsm"].mean("time").assign_attrs(long_name="Land-sea mask"),
    )
