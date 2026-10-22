"""Train the SQF-H models of the paper on the nl domain (see otprof.experiments).

Each run is written to code/runs/models/<experiment>/ (checkpoints, data_pipeline with the fitted scalers,
exp_config.yml, hparams.yaml, TensorBoard events). The published checkpoints from Zenodo go to the same
place; training refuses to overwrite an existing run.

Run from code/:  uv run python scripts/train.py
"""

from __future__ import annotations

import logging

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from otprof import RUNS_ROOT
from otprof.experiments import get_exp

torch.set_float32_matmul_precision("high")


def train(name: str, devices: str | list[int] = "auto", resume_from: str | None = None) -> None:
    """Train experiment `name` into runs/models/<name>/. `resume_from`: checkpoint to continue from."""
    run_dir = RUNS_ROOT / "models" / name
    if run_dir.exists() and resume_from is None:
        raise FileExistsError(f"{run_dir} exists. Move it away or pass `resume_from`.")
    exp = get_exp(name)

    # Setup data loaders for training and validation
    data_train = exp.get_torch_dataset(stage="train")
    model = exp.get_lit_model(data_train)
    loader_train = exp.get_torch_loader(stage="train", data=data_train)
    loader_val = exp.get_torch_loader(stage="val")

    trainer = L.Trainer(
        max_epochs=exp.max_epochs,
        devices=devices,
        logger=TensorBoardLogger(save_dir=RUNS_ROOT, name="models", version=name),
        log_every_n_steps=5,
        callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=5)],
    )
    if trainer.is_global_zero:
        exp.to_disk(run_dir)
    trainer.fit(model, loader_train, loader_val, ckpt_path=resume_from)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train("wrf_pl")  # decoupled training on WRF interpolated to ERA5 pressure levels (the framework)
    train("wrf_native")  # control: native WRF levels in and out
    train("era5_pl_direct")  # ablation: trained on ERA5 PL directly, no decoupling
