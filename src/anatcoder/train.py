"""Lightning + Hydra training entrypoint for Week-2 Vanilla INR baseline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from rich.console import Console
from rich.table import Table

from anatcoder.data.dataset import CTDataModule
from anatcoder.eval.global_metrics import evaluate_reconstruction
from anatcoder.models.network import VanillaINR
from anatcoder.models.ray_utils import compute_near_far_naf
from anatcoder.models.renderer import reconstruct_volume, render_rays
from anatcoder.utils.geometry import CBCTGeometry

console = Console()


class CTReconLitModule(pl.LightningModule):
    """Lightning module for sparse-view CT reconstruction with Vanilla INR."""

    def __init__(self, cfg: DictConfig):
        """Initialize model, geometry-dependent bounds, and training settings."""
        super().__init__()
        self.cfg = cfg
        self.model = VanillaINR(
            encoder_type=str(cfg.model.encoder_type),
            n_levels=int(cfg.model.n_levels),
            n_features_per_level=int(cfg.model.n_features_per_level),
            log2_hashmap_size=int(cfg.model.log2_hashmap_size),
            base_resolution=int(cfg.model.base_resolution),
            per_level_scale=float(cfg.model.per_level_scale),
            n_hidden_layers=int(cfg.model.n_hidden_layers),
            hidden_dim=int(cfg.model.hidden_dim),
            skips=list(getattr(cfg.model, 'skips', [])),
            last_activation=str(getattr(cfg.model, 'last_activation', 'softplus')),
        )

        self.volume_size = [int(v) for v in cfg.data.volume_size]
        self.voxel_size = [float(v) for v in cfg.data.voxel_size]
        self.volume_size_mm = [
            self.volume_size[0] * self.voxel_size[0],
            self.volume_size[1] * self.voxel_size[1],
            self.volume_size[2] * self.voxel_size[2],
        ]
        self.model.volume_size_mm = self.volume_size_mm
        self.use_naf_rays = bool(getattr(cfg.data, 'use_naf_rays', False))
        self.model.bound = float(getattr(cfg.model, 'bound', 0.3)) if self.use_naf_rays else None

        geo = CBCTGeometry(
            DSD=float(cfg.data.geo.DSD),
            DSO=float(cfg.data.geo.DSO),
            n_voxel=self.volume_size,
            d_voxel=self.voxel_size,
            n_detector=list(cfg.data.geo.n_detector),
            d_detector=list(cfg.data.geo.d_detector),
        )
        diag = float(np.linalg.norm(np.asarray(self.volume_size_mm, dtype=np.float32)))
        if self.use_naf_rays:
            self.near, self.far = compute_near_far_naf(geo)
        else:
            self.near = float(geo.DSO - 0.5 * diag)
            self.far = float(geo.DSO + 0.5 * diag)
        self._loss_history: list[float] = []

    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> torch.Tensor:
        """Render predicted line-integral values for input rays."""
        return render_rays(
            model=self.model,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            n_samples=int(self.cfg.train.n_samples),
            near=self.near,
            far=self.far,
            perturb=bool(self.training),
            chunk_size=int(self.cfg.train.chunk_size),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one optimization step with projection-domain MSE loss."""
        _ = batch_idx
        ray_origins = batch['ray_origin'].to(self.device, non_blocking=True)
        ray_directions = batch['ray_direction'].to(self.device, non_blocking=True)
        gt_pixels = batch['gt_pixel'].to(self.device, non_blocking=True)

        pred_pixels = self.forward(ray_origins, ray_directions)
        loss = F.mse_loss(pred_pixels, gt_pixels)
        self._loss_history.append(float(loss.detach().cpu().item()))

        # `self.trainer` raises before attachment; guard using internal reference first.
        if self._trainer is not None and self.trainer is not None:
            self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=gt_pixels.shape[0])
            self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=gt_pixels.shape[0])
            optimizer = self.optimizers(use_pl_optimizer=False)
            if optimizer is not None:
                self.log(
                    'train/lr',
                    float(optimizer.param_groups[0]['lr']),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=gt_pixels.shape[0],
                )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Reconstruct full volume periodically and log PSNR/SSIM/MAE."""
        _ = batch
        if batch_idx != 0:
            return
        if (self.current_epoch + 1) % int(self.cfg.train.val_every_n_epoch) != 0:
            return

        datamodule = self.trainer.datamodule
        if datamodule is None or not hasattr(datamodule, 'get_gt_volume'):
            raise RuntimeError('Validation requires datamodule exposing get_gt_volume().')

        recon = reconstruct_volume(
            model=self.model,
            volume_size=self.volume_size,
            voxel_size=self.voxel_size,
            chunk_size=int(self.cfg.train.recon_chunk_size),
            device=self.device,
        )
        gt = np.asarray(datamodule.get_gt_volume(), dtype=np.float32)
        metrics = evaluate_reconstruction(recon, gt, data_range=1.0)

        self.log('val/psnr', float(metrics['psnr']), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', float(metrics['ssim']), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mae', float(metrics['mae']), on_step=False, on_epoch=True, prog_bar=False)

        if self.logger is not None and hasattr(self.logger, 'experiment'):
            exp = self.logger.experiment
            if hasattr(exp, 'add_image'):
                center_slice = recon[recon.shape[0] // 2 : recon.shape[0] // 2 + 1]
                exp.add_image(
                    'val/recon_axial_center',
                    torch.from_numpy(center_slice),
                    global_step=self.global_step,
                    dataformats='CHW',
                )

    def configure_optimizers(self):
        """Configure Adam with encoder/MLP groups and configurable scheduler."""
        encoder_params = list(self.model.encoder.parameters())
        mlp_params = list(self.model._mlp_layers.parameters()) + list(self.model.attenuation_head.parameters())

        optimizer = torch.optim.Adam(
            [
                {'params': encoder_params, 'lr': float(self.cfg.train.lr_encoder)},
                {'params': mlp_params, 'lr': float(self.cfg.train.lr_mlp)},
            ]
        )

        scheduler_type = str(getattr(self.cfg.train, 'scheduler_type', 'cosine')).lower()
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.cfg.train.step_size),
                gamma=float(self.cfg.train.gamma),
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.cfg.train.max_epochs),
                eta_min=float(self.cfg.train.lr_min),
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def on_train_end(self) -> None:
        """Save final reconstructed volume and print summary metrics."""
        datamodule = self.trainer.datamodule
        if datamodule is None or not hasattr(datamodule, 'get_gt_volume'):
            return

        recon = reconstruct_volume(
            model=self.model,
            volume_size=self.volume_size,
            voxel_size=self.voxel_size,
            chunk_size=int(self.cfg.train.recon_chunk_size),
            device=self.device,
        )
        gt = np.asarray(datamodule.get_gt_volume(), dtype=np.float32)
        metrics = evaluate_reconstruction(recon, gt, data_range=1.0)

        out_dir = Path('outputs') / str(self.cfg.experiment_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'recon_final.npy', recon.astype(np.float32, copy=False))

        table = Table(title='Final Reconstruction Metrics')
        table.add_column('Metric')
        table.add_column('Value', justify='right')
        table.add_row('PSNR', f"{metrics['psnr']:.4f}")
        table.add_row('SSIM', f"{metrics['ssim']:.4f}")
        table.add_row('MAE', f"{metrics['mae']:.6f}")
        console.print(table)


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra-driven training entrypoint."""
    pl.seed_everything(int(cfg.get('seed', 42)), workers=True)

    geo = CBCTGeometry(
        DSD=float(cfg.data.geo.DSD),
        DSO=float(cfg.data.geo.DSO),
        n_voxel=list(cfg.data.volume_size),
        d_voxel=list(cfg.data.voxel_size),
        n_detector=list(cfg.data.geo.n_detector),
        d_detector=list(cfg.data.geo.d_detector),
    )

    case = str(cfg.data.case)
    dm = CTDataModule(
        case_dir=os.path.join(str(cfg.data.data_dir), case),
        proj_dir=os.path.join(str(cfg.data.proj_dir), case),
        n_views=int(cfg.data.n_views),
        geo=geo,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(getattr(cfg.train, 'num_workers', 4)),
        use_naf_rays=bool(getattr(cfg.data, 'use_naf_rays', False)),
    )
    model = CTReconLitModule(cfg)

    callbacks: list[pl_callbacks.Callback] = [
        pl_callbacks.ModelCheckpoint(monitor='val/psnr', mode='max', save_top_k=1),
        pl_callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl_callbacks.RichProgressBar(),
    ]

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=accelerator,
        devices=1,
        precision=str(getattr(cfg.train, 'precision', '32-true')),
        callbacks=callbacks,
        check_val_every_n_epoch=int(cfg.train.val_every_n_epoch),
        logger=pl_loggers.TensorBoardLogger('logs/', name=str(cfg.experiment_name)),
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
