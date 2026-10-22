"""Inference: best checkpoint lookup, model predictions, the WRF PL -> ERA5 PL input swap of the decoupled
model, the Hufnagel-Valley reference profile, and the output quantile mapping (oQM) of ERA5-driven lcn2
predictions."""

from __future__ import annotations

import pathlib

import joblib
import lightning as L
import numpy as np
import torch
import xarray as xr
from research_tools.ot import profiles

from otprof import pipelines
from otprof.experiments import SqueezeFormerExp
from otprof.logging import get_logger
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
