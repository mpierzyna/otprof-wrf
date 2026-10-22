"""Lightning wrapper of the SQF-H model: loss, logging, optimizer."""

import lightning as L
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

from otprof.losses import ddz_loss, edist
from otprof.model import SqueezeformerHModel


def log_sample(tb: SummaryWriter, y_true: torch.Tensor, y_pred: torch.Tensor, epoch: int):
    """Log sample predictions of first batch item to TensorBoard."""
    b, l, f = y_true.shape

    # Plot
    fig, axarr = plt.subplots(ncols=f, nrows=1, figsize=(2 * f, 2), constrained_layout=True)
    if f == 1:
        axarr = [axarr]
    for i, ax in enumerate(axarr):
        ax.plot(y_true[0, :, i].detach().cpu().numpy(), label="true")
        ax.plot(y_pred[0, :, i].detach().cpu().numpy(), label="pred")
        ax.set_title(f"Feature {i}")
        ax.set_xlabel("Level")
        ax.set_ylabel("Value")
        ax.legend()

    tb.add_figure("true_pred_sample", fig, global_step=epoch)
    plt.close(fig)


class LitSqueezeFormerH(L.LightningModule):
    """Height-aware SqueezeFormer (SQF-H). Forcings must be `[z_agl, lz]`: `ddz_loss` uses z_agl,
    the model queries at lz. Training predicts a random sorted subset of `n_query_train` output
    levels per batch so the head learns a function of height instead of the level index;
    val/test/predict use all levels.

    Loss: RMSE of the predictions plus RMSE of the confidence head against the squared error. Energy
    distance and the vertical-derivative error are logged only.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        head_dim: int,
        num_blocks: int,
        num_heads: int,
        conv_kernel_size: int,
        idx_valid: int,
        idx_lz: int,
        use_conf_head: bool = True,
        n_query_train: int = 32,
        lr: float = 1e-3,
        lr_tmax: int = 10,
    ):
        super().__init__()
        self.model = SqueezeformerHModel(
            input_dim=input_dim,
            output_dim=output_dim,
            use_conf_head=use_conf_head,
            embed_dim=embed_dim,
            head_dim=head_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            idx_valid=idx_valid,
            idx_lz=idx_lz,
        )
        # Compiled module: checkpoint keys carry its `_orig_mod.` prefix
        self.model = torch.compile(self.model)
        self.n_query_train = n_query_train
        self.lr = lr
        self.lr_tmax = lr_tmax
        self.save_hyperparameters()

    def forward(self, x, lz_q):
        return self.model(x, lz_q)

    def _shared_step(self, batch, batch_idx):
        x, y_frc, y = batch  # y shape: (B, H, C), y_frc: (B, H, [z_agl, lz])
        if self.training:
            # Random sorted query subset, shared across the batch (static shapes for compile)
            q = torch.randperm(y.shape[1], device=y.device)[: self.n_query_train].sort().values
            y, y_frc = y[:, q], y_frc[:, q]
        y_hat, y_hat_conf = self(x, y_frc[:, :, 1])

        # RMSE
        sq_err_y = (y_hat - y) ** 2
        rmse = torch.sqrt(sq_err_y.mean())

        # Confidence error
        abs_err_conf = torch.abs(y_hat_conf - sq_err_y)
        rmse_conf = torch.sqrt(abs_err_conf.mean())

        loss = rmse + rmse_conf

        edist_y = edist(y_hat, y, normalize=True)
        ddz_mse = ddz_loss(y_hat, y, y_frc[:, :, 0]) / 1e3
        return loss, {
            "mae_loss": torch.abs(y_hat - y).mean(),
            "rmse_loss": rmse,
            "edist": edist_y,
            "ddz_mse": ddz_mse,
        }

    def _log_step(self, stage: str, loss: torch.Tensor, log_metrics: dict) -> None:
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in log_metrics.items():
            self.log(f"{stage}_{k}", v, prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self._log_step("train", loss, log_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self._log_step("val", loss, log_metrics)
        if batch_idx == 0:
            x, y_frc, y = batch
            y_hat, _ = self(x[:2], y_frc[:2, :, 1])
            with torch.no_grad():
                log_sample(self.logger.experiment, y, y_hat, epoch=self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_metrics = self._shared_step(batch, batch_idx)
        self._log_step("test", loss, log_metrics)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y_frc, _ = batch
        return self(x, y_frc[:, :, 1])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]
