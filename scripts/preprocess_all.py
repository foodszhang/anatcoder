"""批量预处理 CT 数据。

Usage:
    python scripts/preprocess_all.py         --input_dir data/raw         --output_dir data/processed         --crop_size 128         --spacing 1.0 1.0 1.0         --hu_window -160 240
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

from anatcoder.data.preprocess import batch_preprocess

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Batch preprocess CT data.')
    parser.add_argument('--input_dir', required=True, help='Raw dataset root directory.')
    parser.add_argument('--output_dir', required=True, help='Processed output directory.')
    parser.add_argument('--crop_size', type=int, default=128, help='Output cubic size.')
    parser.add_argument(
        '--spacing',
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=('SZ', 'SY', 'SX'),
        help='Target spacing in mm.',
    )
    parser.add_argument(
        '--hu_window',
        nargs=2,
        type=float,
        default=(-160.0, 240.0),
        metavar=('HU_MIN', 'HU_MAX'),
        help='HU clipping window.',
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI entrypoint for batch preprocessing."""
    args = parse_args()
    console.print('[bold cyan]Starting CT preprocessing...[/bold cyan]')

    results = batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_spacing=tuple(args.spacing),
        hu_window=tuple(args.hu_window),
        crop_size=args.crop_size,
    )

    table = Table(title='Preprocessing Summary')
    table.add_column('Case')
    table.add_column('Volume Path')
    table.add_column('Segmentation')
    table.add_column('Output Shape')

    for item in results:
        table.add_row(
            str(item.get('case_name', 'unknown')),
            str(item.get('volume_path', '-')),
            'yes' if item.get('seg_path') else 'no',
            str(item.get('output_shape_zyx', '-')),
        )

    console.print(table)
    console.print(f'[green]Done.[/green] Processed {len(results)} case(s).')


if __name__ == '__main__':
    main()
