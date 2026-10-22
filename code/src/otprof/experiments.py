"""SQF-H experiments of the paper: model, data pipeline and training settings.

  wrf_native      WRF native levels -> WRF native levels (regression only, control)
  wrf_pl          WRF interpolated to ERA5 pressure levels -> WRF native levels (decoupled training; at
                  inference fed with ERA5 PL, see otprof.predict.wrf_to_era5)
  era5_pl_direct  ERA5 PL -> WRF native levels (trained on ERA5 directly, no decoupling)

All train on the nl years 2019-2020, validate on 2018 and test on 2017.
"""

from __future__ import annotations

import pathlib

import lightning as L
import torch

import otprof.pipelines as pipelines
from otprof.base import BaseExperiment
from otprof.datasets import ProfileDataset
from otprof.lit import LitSqueezeFormerH
from otprof.logging import get_logger
from otprof.types import TStage

logger = get_logger()

NUM_WORKERS = 16  # data loader workers


class SqueezeFormerExp(BaseExperiment):
    embed_dim: int  # capacity of squeezeformer and attention layers
    head_dim: int  # dimensionality of final prediction head
    num_blocks: int  # depth, ie. number of squeezeformer blocks
    num_heads: int  # number of attention heads
    conv_kernel_size: int  # kernel size of convolutional layers in squeezeformer
    use_conf_head: bool  # use_conf_head

    def get_torch_dataset(self, stage: TStage) -> ProfileDataset:
        ds_X, ds_y = self.dp.split(transform=True, stage=stage)
        return ProfileDataset(
            ds_X=ds_X,
            ds_y=ds_y,
            vars_X=self.features,
            vars_y=self.targets,
            vars_y_frc=self.forcings,
        )

    def get_torch_loader(self, stage: TStage, data: ProfileDataset | None = None) -> torch.utils.data.DataLoader:
        data = data or self.get_torch_dataset(stage=stage)
        shuffle = stage == "train"

        return torch.utils.data.DataLoader(
            data,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

    def get_lit_model(self, data_train: ProfileDataset) -> L.LightningModule:
        logger.info(f"Input levels: {data_train.n_levels_X}, Output levels: {data_train.n_levels_y}")
        return LitSqueezeFormerH(
            input_dim=data_train.n_vars_X,
            output_dim=data_train.n_vars_y,
            use_conf_head=self.use_conf_head,
            embed_dim=self.embed_dim,
            head_dim=self.head_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            conv_kernel_size=self.conv_kernel_size,
            # ProfileDataset reorders variables (surface first), so look up by name
            idx_valid=data_train.vars_X.index("valid"),
            idx_lz=data_train.vars_X.index("lz"),
            lr=1e-3,
            lr_tmax=self.max_epochs,
        )


def get_exp(name: str, domain: str = "nl", scaler_dir: str | pathlib.Path | None = None) -> SqueezeFormerExp:
    """Get experiment by name. Keep experiments in function to avoid pipelines loading on import.

    `domain` selects which dataset/data/<domain>/ tree the pipeline reads (default "nl", the
    trained domain). `scaler_dir` (a saved `data_pipeline` dir) loads trained scalers before
    setup instead of fitting them on `domain` -- required for out-of-domain inference, whose train
    split may be empty (canaries is 2017 = nl test year).
    """
    # Settings shared by all experiments
    base = dict(
        # Features / Target
        features=[
            "z_agl",
            "p",
            "u",
            "v",
            "th",
            # Gradients
            "S",
            "dth_dz",
            # Surface
            "ust",
            "hfx",
            "lh",
            "blh",
            "u10",
            "v10",
            "tk2",
            "slp",
            "lsm",
            # Time features
            "hr_sin",
            "hr_cos",
            "doy_sin",
            "doy_cos",
            # surface estimation of CT2
            "lct2_w71f",
            # Height features (SQF-H)
            "valid",
            "lz",
            "z_sfc",
            "lp_sfc",
        ],
        forcings=["z_agl", "lz"],  # z_agl first: ddz_loss reads y_frc[:, :, 0]; the model queries at lz
        targets=[
            "lcn2",
            "QKE",
            "TSQ",
            "EL_PBL",
        ],
        # Model settings
        use_conf_head=True,
        conv_kernel_size=3,
        embed_dim=48,
        num_blocks=4,
        num_heads=2,
        head_dim=512,
        # Training settings
        batch_size=23 * 1024,
        max_epochs=600,
    )

    if name == "wrf_native":
        dp = pipelines.wrf_wrf.get_wrf_orig_h(domain)
        update = {"name": "sqfh_wrfnative_wrf", "batch_size": 8 * 1024, "max_epochs": 300}
    elif name == "wrf_pl":
        dp = pipelines.wrf_wrf.get_wrfpl_wrf_h(domain)
        update = {"name": "sqfh_wrfpl_wrf"}
    elif name == "era5_pl_direct":
        dp = pipelines.era5_wrf.get_era5pl_wrf_h(domain)
        update = {"name": "sqfh_era5pl_wrf_direct"}
    else:
        raise ValueError(f"Unknown experiment name: {name}")

    if scaler_dir is not None:
        dp.load_ood_scalers(scaler_dir)  # setup() then skips fitting
    return SqueezeFormerExp(dp=dp.setup(), **{**base, **update})
