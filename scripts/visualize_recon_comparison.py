"""Visualize sparse-view reconstruction comparison slices and error maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create 10-view reconstruction comparison figure.')
    parser.add_argument('--case', default='case001', help='Case id under data/processed.')
    parser.add_argument('--views', type=int, default=10, help='Sparse-view count for FDK baseline path.')
    parser.add_argument(
        '--vanilla_experiment',
        default='vanilla_case001_10v_100ep',
        help='Experiment name for Vanilla recon under outputs/<exp>/recon_final.npy',
    )
    parser.add_argument(
        '--best_anatomy_experiment',
        required=True,
        help='Experiment name for best anatomy-prior method recon.',
    )
    parser.add_argument(
        '--best_anatomy_label',
        default='Best anatomy',
        help='Display name of the best anatomy-prior method.',
    )
    parser.add_argument(
        '--output',
        default='results/recon_comparison_10v.png',
        help='Output image path.',
    )
    return parser.parse_args()


def _load_volume(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'{name} not found: {path}')
    vol = np.asarray(np.load(path), dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f'{name} must be 3D, got shape={vol.shape}')
    return vol


def main() -> None:
    args = _parse_args()

    gt = _load_volume(Path('data') / 'processed' / args.case / 'volume.npy', 'GT volume')
    fdk = _load_volume(
        Path('data') / 'projections' / args.case / f'{int(args.views)}views' / 'fdk_recon.npy',
        'FDK reconstruction',
    )
    vanilla = _load_volume(Path('outputs') / args.vanilla_experiment / 'recon_final.npy', 'Vanilla reconstruction')
    best = _load_volume(
        Path('outputs') / args.best_anatomy_experiment / 'recon_final.npy',
        'Best anatomy reconstruction',
    )

    mid = gt.shape[0] // 2
    gt_slice = gt[mid]
    fdk_slice = fdk[mid]
    vanilla_slice = vanilla[mid]
    best_slice = best[mid]

    diff_fdk = np.abs(fdk_slice - gt_slice)
    diff_vanilla = np.abs(vanilla_slice - gt_slice)
    diff_best = np.abs(best_slice - gt_slice)

    img_vmin = float(np.min(gt_slice))
    img_vmax = float(np.max(gt_slice))
    diff_vmax = float(max(np.max(diff_fdk), np.max(diff_vanilla), np.max(diff_best), 1e-6))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    top_imgs = [gt_slice, fdk_slice, vanilla_slice, best_slice]
    top_titles = ['GT', f'FDK ({args.views}v)', 'Vanilla', args.best_anatomy_label]
    for col, (img, title) in enumerate(zip(top_imgs, top_titles)):
        ax = axes[0, col]
        im = ax.imshow(img, cmap='gray', vmin=img_vmin, vmax=img_vmax)
        ax.set_title(title)
        ax.axis('off')
        if col == 3:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    bottom_imgs = [np.zeros_like(gt_slice), diff_fdk, diff_vanilla, diff_best]
    bottom_titles = ['|GT-GT|', '|FDK-GT|', '|Vanilla-GT|', f'|{args.best_anatomy_label}-GT|']
    for col, (img, title) in enumerate(zip(bottom_imgs, bottom_titles)):
        ax = axes[1, col]
        im = ax.imshow(img, cmap='hot', vmin=0.0, vmax=diff_vmax)
        ax.set_title(title)
        ax.axis('off')
        if col == 3:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

