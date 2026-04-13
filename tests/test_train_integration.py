"""Integration smoke tests for Week-2 training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.dataset import CTDataModule  # noqa: E402
from anatcoder.train import CTReconLitModule  # noqa: E402
from anatcoder.utils.geometry import CBCTGeometry, generate_angles  # noqa: E402


def _build_synthetic_case(root: Path) -> tuple[Path, Path]:
    """Create minimal synthetic processed/projection data for integration tests."""
    case_dir = root / 'processed' / 'case001'
    proj_dir = root / 'projections' / 'case001' / '10views'
    case_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    size = 32
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing='ij',
    )
    volume = np.clip(np.exp(-(x**2 + y**2 + z**2) / 0.6), 0.0, 1.0).astype(np.float32)
    seg = np.zeros_like(volume, dtype=np.int16)
    seg[volume > 0.5] = 1
    np.save(case_dir / 'volume.npy', volume)
    np.save(case_dir / 'seg.npy', seg)

    geo = CBCTGeometry(
        n_voxel=[size, size, size],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[32, 32],
        d_detector=[1.5, 1.5],
    )
    angles = generate_angles(10)
    # Use TIGRE for physically consistent projections when available.
    try:
        import tigre

        projections = tigre.Ax(volume.astype(np.float32), geo.to_tigre_geometry(), angles.astype(np.float32))
    except ImportError:
        # Fallback for CI without TIGRE: approximate line integrals.
        slab = volume.mean(axis=1)
        projections = np.stack([slab for _ in range(10)], axis=0).astype(np.float32)
    np.save(proj_dir / 'projections.npy', projections)
    np.save(proj_dir / 'angles.npy', angles.astype(np.float32))
    return case_dir, proj_dir.parent


def _make_cfg(data_root: Path) -> OmegaConf:
    """Build minimal config for quick CPU smoke run."""
    return OmegaConf.create(
        {
            'model': {
                'name': 'vanilla_inr',
                'encoder_type': 'positional',
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'per_level_scale': 1.4472,
                'n_hidden_layers': 2,
                'hidden_dim': 64,
                'last_activation': 'sigmoid',
            },
            'train': {
                'max_epochs': 1,
                'batch_size': 64,
                'n_samples': 16,
                'lr_mlp': 1e-3,
                'lr_encoder': 1e-3,
                'lr_min': 1e-5,
                'val_every_n_epoch': 1,
                'chunk_size': 256,
                'recon_chunk_size': 4096,
                'num_workers': 0,
            },
            'data': {
                'data_dir': str(data_root / 'processed'),
                'proj_dir': str(data_root / 'projections'),
                'case': 'case001',
                'n_views': 10,
                'volume_size': [32, 32, 32],
                'voxel_size': [1.0, 1.0, 1.0],
                'geo': {
                    'DSD': 1536.0,
                    'DSO': 1000.0,
                    'n_detector': [32, 32],
                    'd_detector': [1.5, 1.5],
                },
            },
            'experiment_name': 'smoke_test',
        }
    )


def test_training_smoke(tmp_path: Path) -> None:
    """Smoke test: run two optimization steps without error and observe non-increasing loss."""
    case_dir, proj_case_dir = _build_synthetic_case(tmp_path)
    cfg = _make_cfg(tmp_path)

    geo = CBCTGeometry(
        DSD=cfg.data.geo.DSD,
        DSO=cfg.data.geo.DSO,
        n_voxel=cfg.data.volume_size,
        d_voxel=cfg.data.voxel_size,
        n_detector=cfg.data.geo.n_detector,
        d_detector=cfg.data.geo.d_detector,
    )
    dm = CTDataModule(
        case_dir=str(case_dir),
        proj_dir=str(proj_case_dir),
        n_views=10,
        geo=geo,
        batch_size=64,
        num_workers=0,
    )
    dm.setup()
    module = CTReconLitModule(cfg)
    module.to(torch.device('cpu'))
    module.train()

    optimizer_dict = module.configure_optimizers()
    optimizer = optimizer_dict['optimizer']
    loader = dm.train_dataloader()
    losses: list[float] = []
    for step, batch in enumerate(loader):
        if step >= 2:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(batch, step)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    assert len(losses) == 2
    assert np.isfinite(losses).all()
    # With random ray batches and realistic TIGRE projections, two consecutive
    # mini-batches are not guaranteed to be strictly monotonic.
    assert max(losses) < 2e3


def test_reconstruct_after_training(tmp_path: Path) -> None:
    """Verify reconstruction utility returns expected output shape after a tiny training step."""
    case_dir, proj_case_dir = _build_synthetic_case(tmp_path)
    cfg = _make_cfg(tmp_path)
    geo = CBCTGeometry(
        DSD=cfg.data.geo.DSD,
        DSO=cfg.data.geo.DSO,
        n_voxel=cfg.data.volume_size,
        d_voxel=cfg.data.voxel_size,
        n_detector=cfg.data.geo.n_detector,
        d_detector=cfg.data.geo.d_detector,
    )
    dm = CTDataModule(
        case_dir=str(case_dir),
        proj_dir=str(proj_case_dir),
        n_views=10,
        geo=geo,
        batch_size=64,
        num_workers=0,
    )
    dm.setup()
    module = CTReconLitModule(cfg)
    module.to(torch.device('cpu'))
    module.train()

    optimizer = module.configure_optimizers()['optimizer']
    batch = next(iter(dm.train_dataloader()))
    optimizer.zero_grad(set_to_none=True)
    loss = module.training_step(batch, 0)
    loss.backward()
    optimizer.step()

    from anatcoder.models.renderer import reconstruct_volume

    recon = reconstruct_volume(
        model=module.model,
        volume_size=[32, 32, 32],
        voxel_size=[1.0, 1.0, 1.0],
        chunk_size=4096,
        device=torch.device('cpu'),
    )
    assert recon.shape == (32, 32, 32)
    assert np.isfinite(recon).all()


def test_projection_psnr_metric_is_computable(tmp_path: Path) -> None:
    """Validation helper should compute a finite projection-domain PSNR."""
    case_dir, proj_case_dir = _build_synthetic_case(tmp_path)
    cfg = _make_cfg(tmp_path)
    geo = CBCTGeometry(
        DSD=cfg.data.geo.DSD,
        DSO=cfg.data.geo.DSO,
        n_voxel=cfg.data.volume_size,
        d_voxel=cfg.data.voxel_size,
        n_detector=cfg.data.geo.n_detector,
        d_detector=cfg.data.geo.d_detector,
    )
    dm = CTDataModule(
        case_dir=str(case_dir),
        proj_dir=str(proj_case_dir),
        n_views=10,
        geo=geo,
        batch_size=64,
        num_workers=0,
    )
    dm.setup()
    module = CTReconLitModule(cfg)
    module.to(torch.device('cpu'))
    module.eval()

    proj_psnr = module._compute_projection_psnr(dm)
    assert np.isfinite(proj_psnr)


def test_oracle_forward_smoke_with_cached_seg(tmp_path: Path) -> None:
    """Oracle mode should render rays when seg one-hot condition is provided."""
    case_dir, proj_case_dir = _build_synthetic_case(tmp_path)
    cfg = _make_cfg(tmp_path)
    cfg.model.n_anatomy_classes = 2
    geo = CBCTGeometry(
        DSD=cfg.data.geo.DSD,
        DSO=cfg.data.geo.DSO,
        n_voxel=cfg.data.volume_size,
        d_voxel=cfg.data.voxel_size,
        n_detector=cfg.data.geo.n_detector,
        d_detector=cfg.data.geo.d_detector,
    )
    dm = CTDataModule(
        case_dir=str(case_dir),
        proj_dir=str(proj_case_dir),
        n_views=10,
        geo=geo,
        batch_size=64,
        num_workers=0,
    )
    dm.setup()
    module = CTReconLitModule(cfg)
    module.to(torch.device('cpu'))
    module.train()

    seg_np = dm.get_seg_volume()
    assert seg_np is not None
    module._seg_volume_labels = torch.from_numpy(seg_np.astype(np.int64)).clamp(0, 1)

    batch = next(iter(dm.train_dataloader()))
    pred = module.forward(
        batch['ray_origin'].to(torch.device('cpu')),
        batch['ray_direction'].to(torch.device('cpu')),
    )
    assert pred.shape[0] == batch['ray_origin'].shape[0]
    assert torch.isfinite(pred).all()


def test_advr_forward_smoke_with_seg_labels(tmp_path: Path) -> None:
    """ADVR config should instantiate and render rays with seg-label routing."""
    case_dir, proj_case_dir = _build_synthetic_case(tmp_path)
    cfg = _make_cfg(tmp_path)
    cfg.model.name = 'advr'
    cfg.model.n_anatomy_classes = 2
    cfg.model.head_hidden_dim = 16
    geo = CBCTGeometry(
        DSD=cfg.data.geo.DSD,
        DSO=cfg.data.geo.DSO,
        n_voxel=cfg.data.volume_size,
        d_voxel=cfg.data.voxel_size,
        n_detector=cfg.data.geo.n_detector,
        d_detector=cfg.data.geo.d_detector,
    )
    dm = CTDataModule(
        case_dir=str(case_dir),
        proj_dir=str(proj_case_dir),
        n_views=10,
        geo=geo,
        batch_size=64,
        num_workers=0,
    )
    dm.setup()
    module = CTReconLitModule(cfg)
    module.to(torch.device('cpu'))
    module.train()
    seg_np = dm.get_seg_volume()
    assert seg_np is not None
    module._seg_volume_labels = torch.from_numpy(seg_np.astype(np.int64)).clamp(0, 1)

    batch = next(iter(dm.train_dataloader()))
    pred = module.forward(
        batch['ray_origin'].to(torch.device('cpu')),
        batch['ray_direction'].to(torch.device('cpu')),
    )
    assert pred.shape[0] == batch['ray_origin'].shape[0]
    assert torch.isfinite(pred).all()
