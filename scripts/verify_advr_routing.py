"""Verify ADVR routing and segmentation-conditioned rendering behavior."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.advr_network import ADVRNetwork  # noqa: E402
from anatcoder.models.renderer import reconstruct_volume, render_rays  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify ADVR routing correctness.')
    parser.add_argument('--checkpoint', default=None, help='Path to ADVR Lightning checkpoint (.ckpt).')
    parser.add_argument('--model_config', default='configs/model/advr.yaml', help='ADVR model config YAML.')
    parser.add_argument('--processed_dir', default='data/processed', help='Processed dataset root.')
    parser.add_argument('--case', default='case001', help='Case id under processed_dir.')
    parser.add_argument('--n_points_per_class', type=int, default=100, help='Random points sampled per class.')
    parser.add_argument('--n_anatomy_classes', type=int, default=10, help='Anatomy class count.')
    parser.add_argument('--voxel_size_mm', type=float, default=1.0, help='Voxel size (mm) for recon grid.')
    parser.add_argument('--recon_chunk_size', type=int, default=65536, help='Chunk size for reconstruct_volume.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', default=None, help='Device string, e.g. cuda or cpu.')
    return parser.parse_args()


def _find_latest_checkpoint() -> Path:
    candidates = sorted(
        Path('logs').glob('advr*/version_*/checkpoints/*.ckpt'),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError('No ADVR checkpoint found under logs/advr*/version_*/checkpoints/*.ckpt')
    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        name = path.name
        epoch_match = re.search(r'epoch=(\d+)', name)
        step_match = re.search(r'step=(\d+)', name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        step = int(step_match.group(1)) if step_match else -1
        scored.append((epoch, step, path))
    scored.sort(key=lambda item: (item[0], item[1], item[2].stat().st_mtime))
    return scored[-1][2]


def _load_advr_model(checkpoint: Path, model_cfg_path: Path, n_classes: int, device: torch.device) -> ADVRNetwork:
    model_cfg = OmegaConf.load(model_cfg_path)
    model = ADVRNetwork(
        encoder_type=str(model_cfg.encoder_type),
        n_levels=int(model_cfg.n_levels),
        n_features_per_level=int(model_cfg.n_features_per_level),
        log2_hashmap_size=int(model_cfg.log2_hashmap_size),
        base_resolution=int(model_cfg.base_resolution),
        per_level_scale=float(model_cfg.per_level_scale),
        n_hidden_layers=int(model_cfg.n_hidden_layers),
        hidden_dim=int(model_cfg.hidden_dim),
        head_hidden_dim=int(model_cfg.head_hidden_dim),
        last_activation=str(getattr(model_cfg, 'last_activation', 'sigmoid')),
        n_anatomy_classes=int(n_classes),
    )

    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    state_dict = checkpoint_data.get('state_dict', checkpoint_data)
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            model_state[key[len('model.') :]] = value
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model.to(device)


def _voxel_ijk_to_normalized(ijk: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    d, h, w = shape
    z = (ijk[:, 0].to(torch.float32) + 0.5) / float(d)
    y = (ijk[:, 1].to(torch.float32) + 0.5) / float(h)
    x = (ijk[:, 2].to(torch.float32) + 0.5) / float(w)
    return torch.stack([z, y, x], dim=-1)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint()
    model_cfg_path = Path(args.model_config)

    seg_path = Path(args.processed_dir) / args.case / 'seg.npy'
    if not seg_path.exists():
        raise FileNotFoundError(f'seg.npy not found: {seg_path}')
    seg_np = np.asarray(np.load(seg_path), dtype=np.int64)
    seg_t = torch.from_numpy(seg_np).to(device=device, dtype=torch.long)
    shape = tuple(int(v) for v in seg_np.shape)
    if len(shape) != 3:
        raise ValueError(f'seg.npy must be 3D, got shape={shape}')

    model = _load_advr_model(
        checkpoint=ckpt_path,
        model_cfg_path=model_cfg_path,
        n_classes=int(args.n_anatomy_classes),
        device=device,
    )
    model.voxel_size_world = [float(args.voxel_size_mm) / 1000.0] * 3
    model.volume_size_world = [
        float(shape[0]) * model.voxel_size_world[0],
        float(shape[1]) * model.voxel_size_world[1],
        float(shape[2]) * model.voxel_size_world[2],
    ]
    model.zero_outside_volume = True
    model.line_integral_scale = 1000.0

    print(f'checkpoint: {ckpt_path}')
    print(f'device: {device}')
    print(f'seg shape: {shape}, classes in seg: {sorted(int(v) for v in np.unique(seg_np))}')

    # Check 1: correct head vs wrong head response difference.
    print('\n[CHECK 1] head routing separation')
    check1_ok = True
    with torch.no_grad():
        for class_idx in range(int(args.n_anatomy_classes)):
            coords_all = torch.nonzero(seg_t == class_idx, as_tuple=False)
            if coords_all.shape[0] == 0:
                print(f'class {class_idx}: skipped (no voxels)')
                continue
            n_pick = min(int(args.n_points_per_class), int(coords_all.shape[0]))
            sample_idx = torch.randint(0, int(coords_all.shape[0]), (n_pick,), device=device)
            ijk = coords_all[sample_idx]
            coords = _voxel_ijk_to_normalized(ijk, shape=shape).to(device=device)
            correct_labels = torch.full((n_pick,), class_idx, dtype=torch.long, device=device)
            wrong_labels = torch.full(
                (n_pick,),
                (class_idx + 1) % int(args.n_anatomy_classes),
                dtype=torch.long,
                device=device,
            )

            mu_correct = model.query_density(coords, anatomy_labels=correct_labels).squeeze(-1)
            mu_wrong = model.query_density(coords, anatomy_labels=wrong_labels).squeeze(-1)
            mean_correct = float(mu_correct.mean().item())
            mean_wrong = float(mu_wrong.mean().item())
            mean_abs_diff = float((mu_correct - mu_wrong).abs().mean().item())
            print(
                f'class {class_idx}: correct={mean_correct:.6f}, wrong={mean_wrong:.6f}, '
                f'|diff|={mean_abs_diff:.6f}'
            )
            if mean_abs_diff < 1e-3:
                check1_ok = False

    # Check 3: seg sampling coords vs network query coords consistency.
    print('\n[CHECK 3] seg-sampling coords vs model-query coords')
    debug_capture: dict[str, object] = {}
    ray_origins = torch.zeros((16, 3), dtype=torch.float32, device=device)
    ray_dirs = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(16, 1)
    half_depth = 0.5 * model.volume_size_world[0]
    _ = render_rays(
        model=model,
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        n_samples=64,
        near=-half_depth,
        far=half_depth,
        perturb=False,
        chunk_size=4096,
        seg_volume=seg_t,
        n_anatomy_classes=int(args.n_anatomy_classes),
        debug_capture=debug_capture,
    )
    seg_preview = debug_capture.get('seg_sample_coords_preview')
    query_preview = debug_capture.get('network_query_coords_preview')
    same_storage = bool(debug_capture.get('coord_same_storage', False))
    coord_max_abs_diff = float(debug_capture.get('coord_max_abs_diff', float('inf')))
    if isinstance(seg_preview, torch.Tensor):
        print(f'seg coords preview: {seg_preview.tolist()}')
    if isinstance(query_preview, torch.Tensor):
        print(f'query coords preview: {query_preview.tolist()}')
    print(f'coord_same_storage={same_storage}, coord_max_abs_diff={coord_max_abs_diff:.3e}')
    check3_ok = same_storage and coord_max_abs_diff <= 1e-8

    # Check 4: reconstruct class-wise statistics.
    print('\n[CHECK 4] class-wise recon stats')
    recon = reconstruct_volume(
        model=model,
        volume_size=[shape[0], shape[1], shape[2]],
        voxel_size=[float(args.voxel_size_mm)] * 3,
        chunk_size=int(args.recon_chunk_size),
        device=device,
        seg_volume=seg_t,
        n_anatomy_classes=int(args.n_anatomy_classes),
    )
    means: list[float] = []
    stds: list[float] = []
    for class_idx in range(int(args.n_anatomy_classes)):
        mask = seg_np == class_idx
        if not np.any(mask):
            print(f'class {class_idx}: skipped (no voxels)')
            continue
        vals = recon[mask]
        class_mean = float(vals.mean())
        class_std = float(vals.std())
        means.append(class_mean)
        stds.append(class_std)
        print(f'class {class_idx}: mean={class_mean:.4f}, std={class_std:.4f}, n={int(mask.sum())}')
    if means:
        print(f'mean spread std={float(np.std(means)):.6f}, std spread std={float(np.std(stds)):.6f}')
    check4_ok = bool(means) and not (float(np.std(means)) < 1e-3 and float(np.std(stds)) < 1e-3)

    all_ok = check1_ok and check3_ok and check4_ok
    print('\nCHECK SUMMARY:')
    print(f'check1_routing={check1_ok}')
    print(f'check3_coord_consistency={check3_ok}')
    print(f'check4_recon_separation={check4_ok}')
    if all_ok:
        print('ADVR_VERIFIED_CORRECT_BUT_INEFFECTIVE')
    else:
        print('ADVR_VERIFICATION_FOUND_POTENTIAL_BUG')


if __name__ == '__main__':
    main()
