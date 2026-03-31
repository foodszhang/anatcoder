"""生成稀疏视角投影。

Usage:
    python scripts/generate_projections.py         --data_dir data/processed         --output_dir data/projections         --n_views 50 20 10         --volume_size 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.projection import generate_sparse_projections
from anatcoder.utils.geometry import CBCTGeometry

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate sparse-view projections with TIGRE.')
    parser.add_argument('--data_dir', required=True, help='Processed data root with case folders.')
    parser.add_argument('--output_dir', required=True, help='Projection output root directory.')
    parser.add_argument('--n_views', nargs='+', type=int, required=True, help='View counts, e.g. 50 20 10.')
    parser.add_argument('--volume_size', type=int, default=128, help='Volume side length in voxels.')
    parser.add_argument(
        '--detector_size',
        type=int,
        default=None,
        help='Detector side length in pixels. Defaults to volume_size * 2.',
    )
    parser.add_argument('--DSD', type=float, default=1536.0, help='Source-to-detector distance (mm).')
    parser.add_argument('--DSO', type=float, default=1000.0, help='Source-to-object distance (mm).')
    return parser.parse_args()


def main() -> None:
    """Run sparse projection generation for all processed cases."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    detector_size = args.detector_size if args.detector_size is not None else args.volume_size * 2
    geometry = CBCTGeometry(
        DSD=args.DSD,
        DSO=args.DSO,
        n_voxel=[args.volume_size, args.volume_size, args.volume_size],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[detector_size, detector_size],
        d_detector=[1.5, 1.5],
    )

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f'No case directories found under: {data_dir}')

    summary_rows: list[tuple[str, str, str, str, str]] = []
    failures: list[str] = []

    for case_dir in case_dirs:
        volume_path = case_dir / 'volume.npy'
        if not volume_path.exists():
            console.print(f'[yellow]Skipping {case_dir.name}[/yellow]: missing volume.npy')
            continue

        try:
            result = generate_sparse_projections(
                volume_path=str(volume_path),
                n_views_list=[int(v) for v in args.n_views],
                output_dir=str(args.output_dir),
                geo=geometry,
            )
            for n_views, payload in result['results'].items():
                metrics = payload['metrics']
                summary_rows.append(
                    (
                        result['case_name'],
                        f'{n_views}views',
                        f"{metrics['psnr']:.3f}",
                        f"{metrics['ssim']:.4f}",
                        f"{metrics['mae']:.6f}",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            message = f'{case_dir.name}: {exc}'
            failures.append(message)
            console.print(f'[red]Failed[/red] {message}')

    if summary_rows:
        table = Table(title='FDK Summary')
        table.add_column('Case')
        table.add_column('Views')
        table.add_column('PSNR')
        table.add_column('SSIM')
        table.add_column('MAE')
        for row in summary_rows:
            table.add_row(*row)
        console.print(table)

    if failures:
        details = "\n".join(f"- {f}" for f in failures)
        raise RuntimeError(f"Projection generation failed for some cases:\n{details}")

    if not summary_rows:
        raise RuntimeError('No projection outputs were generated. Check input data directory.')

    console.print('[green]Projection generation finished successfully.[/green]')


if __name__ == '__main__':
    main()
