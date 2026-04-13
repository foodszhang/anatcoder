"""Lightning + Hydra training entrypoint for Week-2 Vanilla INR baseline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from anatcoder.data.dataset import CTDataModule
from anatcoder.eval.global_metrics import compute_psnr, evaluate_reconstruction
from anatcoder.losses.anatomy_tv import anatomy_tv_loss
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
        self.model_name = str(getattr(cfg.model, 'name', 'vanilla_inr')).lower()
        self.n_anatomy_classes = int(getattr(cfg.model, 'n_anatomy_classes', 0))
        if self.model_name == 'advr':
            from anatcoder.models.advr_network import ADVRNetwork

            self.model = ADVRNetwork(
                encoder_type=str(cfg.model.encoder_type),
                n_levels=int(cfg.model.n_levels),
                n_features_per_level=int(cfg.model.n_features_per_level),
                log2_hashmap_size=int(cfg.model.log2_hashmap_size),
                base_resolution=int(cfg.model.base_resolution),
                per_level_scale=float(cfg.model.per_level_scale),
                n_hidden_layers=int(cfg.model.n_hidden_layers),
                hidden_dim=int(cfg.model.hidden_dim),
                head_hidden_dim=int(cfg.model.head_hidden_dim),
                last_activation=str(getattr(cfg.model, 'last_activation', 'sigmoid')),
                n_anatomy_classes=self.n_anatomy_classes,
                skips=list(getattr(cfg.model, 'skips', [])),
            )
        else:
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
                n_anatomy_classes=self.n_anatomy_classes,
            )

        self.volume_size = [int(v) for v in cfg.data.volume_size]
        self.voxel_size = [float(v) for v in cfg.data.voxel_size]
        self.volume_size_mm = [
            self.volume_size[0] * self.voxel_size[0],
            self.volume_size[1] * self.voxel_size[1],
            self.volume_size[2] * self.voxel_size[2],
        ]
        self.volume_size_m = [v / 1000.0 for v in self.volume_size_mm]
        self.model.volume_size_mm = self.volume_size_mm
        self.model.bound = None
        self.model.volume_size_world = self.volume_size_m
        self.model.voxel_size_world = [v / 1000.0 for v in self.voxel_size]
        self.model.coord_axis_mode = 'identity'
        self.model.zero_outside_volume = True
        self.model.line_integral_scale = 1000.0

        geo = CBCTGeometry(
            DSD=float(cfg.data.geo.DSD),
            DSO=float(cfg.data.geo.DSO),
            n_voxel=self.volume_size,
            d_voxel=self.voxel_size,
            n_detector=list(cfg.data.geo.n_detector),
            d_detector=list(cfg.data.geo.d_detector),
        )
        self.near, self.far = compute_near_far_naf(geo)
        self._loss_history: list[float] = []
        self._seg_volume_raw: torch.Tensor | None = None
        self._seg_volume_labels: torch.Tensor | None = None

    def _get_seg_volume_raw(self, datamodule: Any | None = None) -> torch.Tensor | None:
        """Get cached raw segmentation labels ``[D,H,W]`` for the current case."""
        if self._seg_volume_raw is not None:
            if self._seg_volume_raw.device != self.device:
                self._seg_volume_raw = self._seg_volume_raw.to(self.device, non_blocking=True)
            return self._seg_volume_raw

        source_dm = datamodule
        if source_dm is None and self._trainer is not None and self.trainer is not None:
            source_dm = self.trainer.datamodule
        if source_dm is None or not hasattr(source_dm, 'get_seg_volume'):
            return None

        seg_np = source_dm.get_seg_volume()
        if seg_np is None:
            return None

        self._seg_volume_raw = torch.from_numpy(np.asarray(seg_np, dtype=np.int64)).to(self.device, non_blocking=True)
        return self._seg_volume_raw

    def _get_seg_volume_labels(self, datamodule: Any | None = None) -> torch.Tensor | None:
        """Get seg labels clamped to configured anatomy classes for model conditioning."""
        if self.n_anatomy_classes <= 0:
            return None
        if self._seg_volume_labels is not None:
            if self._seg_volume_labels.device != self.device:
                self._seg_volume_labels = self._seg_volume_labels.to(self.device, non_blocking=True)
            return self._seg_volume_labels
        raw_labels = self._get_seg_volume_raw(datamodule)
        if raw_labels is None:
            raise RuntimeError('Anatomy-conditioned model requested, but dataset has no seg.npy.')
        self._seg_volume_labels = raw_labels.clamp(0, self.n_anatomy_classes - 1).to(self.device, non_blocking=True)
        return self._seg_volume_labels

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        debug_capture: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Render predicted line-integral values for input rays."""
        seg_volume_labels = self._get_seg_volume_labels() if self.n_anatomy_classes > 0 else None
        return render_rays(
            model=self.model,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            n_samples=int(self.cfg.train.n_samples),
            near=self.near,
            far=self.far,
            perturb=bool(self.training),
            chunk_size=int(self.cfg.train.chunk_size),
            seg_volume=seg_volume_labels,
            n_anatomy_classes=self.n_anatomy_classes,
            debug_capture=debug_capture,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one optimization step with projection-domain MSE loss."""
        ray_origins = batch['ray_origin'].to(self.device, non_blocking=True)
        ray_directions = batch['ray_direction'].to(self.device, non_blocking=True)
        gt_pixels = batch['gt_pixel'].to(self.device, non_blocking=True)

        debug_verify = bool(getattr(self.cfg.train, 'debug_verify_advr', False))
        debug_capture: dict[str, Any] | None = None
        if debug_verify and self.n_anatomy_classes > 0 and self.current_epoch == 0 and batch_idx == 0:
            debug_capture = {}

        pred_pixels = self.forward(ray_origins, ray_directions, debug_capture=debug_capture)
        loss_proj = F.mse_loss(pred_pixels, gt_pixels)

        if debug_capture is not None:
            anatomy_labels = debug_capture.get('anatomy_labels')
            assert anatomy_labels is not None, 'anatomy_labels must not be None for anatomy-conditioned training.'
            assert int(torch.min(anatomy_labels).item()) >= 0, 'anatomy_labels must be non-negative.'
            assert int(torch.max(anatomy_labels).item()) < self.n_anatomy_classes, 'anatomy_labels out of class range.'
            non_background_ratio = float((anatomy_labels > 0).float().mean().item())
            assert non_background_ratio > 0.01, 'anatomy_labels appear to be almost all background.'
            console.print(
                f'[yellow]ADVR_CHECK2[/yellow] label-range=[{int(torch.min(anatomy_labels).item())},'
                f'{int(torch.max(anatomy_labels).item())}] non_bg_ratio={non_background_ratio:.4f}'
            )
            seg_preview = debug_capture.get('seg_sample_coords_preview')
            query_preview = debug_capture.get('network_query_coords_preview')
            same_storage = bool(debug_capture.get('coord_same_storage', False))
            coord_diff = float(debug_capture.get('coord_max_abs_diff', float('nan')))
            console.print(
                '[yellow]ADVR_CHECK3[/yellow] '
                f'coord_same_storage={same_storage} coord_max_abs_diff={coord_diff:.3e}'
            )
            if isinstance(seg_preview, torch.Tensor):
                console.print(f'[yellow]ADVR_CHECK3[/yellow] seg_coords_preview={seg_preview.tolist()}')
            if isinstance(query_preview, torch.Tensor):
                console.print(f'[yellow]ADVR_CHECK3[/yellow] query_coords_preview={query_preview.tolist()}')

        lambda_atv = float(getattr(self.cfg.train, 'lambda_atv', 0.0))
        atv_every_n_steps = max(1, int(getattr(self.cfg.train, 'atv_every_n_steps', 4)))
        loss_tv = torch.zeros((), dtype=loss_proj.dtype, device=self.device)
        if lambda_atv > 0.0 and (batch_idx % atv_every_n_steps == 0):
            seg_for_tv = self._get_seg_volume_raw()
            if seg_for_tv is None:
                raise RuntimeError('Anatomy TV requested, but dataset has no seg.npy.')
            loss_tv = anatomy_tv_loss(
                model=self.model,
                seg_volume=seg_for_tv,
                volume_size_world=self.model.volume_size_world,
                n_sample_pairs=int(getattr(self.cfg.train, 'atv_n_pairs', 4096)),
                alpha=float(getattr(self.cfg.train, 'atv_alpha', 0.05)),
                n_anatomy_classes=self.n_anatomy_classes,
                device=self.device,
            ).to(dtype=loss_proj.dtype)

        loss = loss_proj + lambda_atv * loss_tv
        self._loss_history.append(float(loss.detach().cpu().item()))

        # `self.trainer` raises before attachment; guard using internal reference first.
        if self._trainer is not None and self.trainer is not None:
            self.log(
                'train/loss_proj',
                loss_proj,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=gt_pixels.shape[0],
            )
            self.log(
                'train/loss_tv',
                loss_tv,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=gt_pixels.shape[0],
            )
            self.log(
                'train/loss',
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=gt_pixels.shape[0],
            )
            self.log(
                'train/loss_step',
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=gt_pixels.shape[0],
            )
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
            seg_volume=self._get_seg_volume_labels(datamodule),
            n_anatomy_classes=self.n_anatomy_classes,
        )
        gt = np.asarray(datamodule.get_gt_volume(), dtype=np.float32)
        metrics = evaluate_reconstruction(recon, gt, data_range=1.0)

        self.log('val/psnr', float(metrics['psnr']), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', float(metrics['ssim']), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mae', float(metrics['mae']), on_step=False, on_epoch=True, prog_bar=False)
        proj_psnr = self._compute_projection_psnr(datamodule)
        self.log('val/proj_psnr', float(proj_psnr), on_step=False, on_epoch=True, prog_bar=False)

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

    def _compute_projection_psnr(self, datamodule: Any) -> float:
        """Render one random view and compute projection-domain PSNR against GT."""
        dataset = getattr(datamodule, 'dataset', None)
        if dataset is None:
            raise RuntimeError('Validation datamodule must expose initialized `dataset`.')
        required = ['_ray_origins', '_ray_directions', '_gt_pixels', 'n_view_loaded', 'det_rows', 'det_cols']
        for attr in required:
            if not hasattr(dataset, attr):
                raise RuntimeError(f'Dataset missing required attribute for projection PSNR: {attr}')

        n_views = int(dataset.n_view_loaded)
        rows = int(dataset.det_rows)
        cols = int(dataset.det_cols)
        if n_views <= 0 or rows <= 0 or cols <= 0:
            raise RuntimeError(
                f'Invalid dataset geometry for projection PSNR: views={n_views}, rows={rows}, cols={cols}'
            )

        view_idx = int(np.random.randint(0, n_views))
        rays_per_view = rows * cols
        start = view_idx * rays_per_view
        end = start + rays_per_view

        ray_origins = torch.from_numpy(dataset._ray_origins[start:end]).to(self.device, non_blocking=True)
        ray_directions = torch.from_numpy(dataset._ray_directions[start:end]).to(self.device, non_blocking=True)
        gt_pixels = torch.from_numpy(dataset._gt_pixels[start:end]).to(self.device, non_blocking=True)

        pred_pixels = render_rays(
            model=self.model,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            n_samples=int(self.cfg.train.n_samples),
            near=self.near,
            far=self.far,
            perturb=False,
            chunk_size=int(self.cfg.train.chunk_size),
            seg_volume=self._get_seg_volume_labels(datamodule),
            n_anatomy_classes=self.n_anatomy_classes,
        )
        return float(
            compute_psnr(
                pred_pixels.detach().float().cpu().numpy(),
                gt_pixels.detach().float().cpu().numpy(),
                data_range=max(
                    1e-6,
                    float((gt_pixels.max() - gt_pixels.min()).detach().cpu().item()),
                ),
            )
        )

    def configure_optimizers(self):
        """Configure Adam with encoder/MLP groups and configurable scheduler."""
        encoder_params = list(self.model.encoder.parameters()) if hasattr(self.model, 'encoder') else []
        encoder_ids = {id(p) for p in encoder_params}
        mlp_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': float(self.cfg.train.lr_encoder)})
        if mlp_params:
            param_groups.append({'params': mlp_params, 'lr': float(self.cfg.train.lr_mlp)})
        if not param_groups:
            raise RuntimeError('Model has no trainable parameters.')
        optimizer = torch.optim.Adam(param_groups)

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
            seg_volume=self._get_seg_volume_labels(datamodule),
            n_anatomy_classes=self.n_anatomy_classes,
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
    torch.set_float32_matmul_precision(str(getattr(cfg.train, 'matmul_precision', 'high')))

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
    )
    model = CTReconLitModule(cfg)

    interactive_progress = bool(getattr(cfg.train, 'enable_progress_bar', True)) and sys.stdout.isatty()

    callbacks: list[pl_callbacks.Callback] = [
        pl_callbacks.ModelCheckpoint(monitor='val/psnr', mode='max', save_top_k=1),
        pl_callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]
    if interactive_progress:
        callbacks.append(pl_callbacks.RichProgressBar())

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=accelerator,
        devices=1,
        precision=str(getattr(cfg.train, 'precision', '32-true')),
        callbacks=callbacks,
        check_val_every_n_epoch=int(cfg.train.val_every_n_epoch),
        logger=pl_loggers.TensorBoardLogger('logs/', name=str(cfg.experiment_name)),
        enable_progress_bar=interactive_progress,
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
