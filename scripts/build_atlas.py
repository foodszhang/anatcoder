"""构建解剖 Atlas。

Usage:
    # Oracle atlas（Week 1）
    python scripts/build_atlas.py --mode oracle         --seg_path data/processed/case001/seg.npy         --output data/atlas/oracle_atlas.npy
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.atlas import AtlasBuilder
from anatcoder.utils.io import save_numpy

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for atlas construction."""
    parser = argparse.ArgumentParser(description='Build anatomy atlas.')
    parser.add_argument('--mode', choices=['oracle', 'population'], required=True)
    parser.add_argument('--seg_path', nargs='+', help='Path or glob to segmentation .npy file(s).')
    parser.add_argument('--seg_dir', help='Population mode input directory.')
    parser.add_argument('--reference_path', help='Population mode reference path.')
    parser.add_argument('--n_cases', type=int, default=50)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_classes', type=int, default=105)
    return parser.parse_args()


def _resolve_seg_paths(seg_args: list[str] | None) -> list[Path]:
    """Resolve segmentation inputs from args with glob support."""
    if not seg_args:
        return []
    resolved: list[Path] = []
    for raw in seg_args:
        matches = [Path(p) for p in glob.glob(raw)]
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(Path(raw))
    unique = sorted(set(resolved))
    return unique


def main() -> None:
    """Run atlas building entrypoint."""
    args = parse_args()

    if args.mode == 'oracle':
        seg_paths = _resolve_seg_paths(args.seg_path)
        if not seg_paths:
            raise ValueError('--seg_path is required for oracle mode')

        if len(seg_paths) > 1:
            console.print(
                f'[yellow]Multiple seg paths provided ({len(seg_paths)}). '
                f'Using the first one: {seg_paths[0]}[/yellow]'
            )
        seg_path = seg_paths[0]

        atlas = AtlasBuilder.from_oracle(seg_path=seg_path, n_classes=args.n_classes)
        save_numpy(atlas, args.output)
        console.print(
            f'[green]Oracle atlas saved[/green]: {args.output} '
            f'(shape={tuple(atlas.shape)}, dtype={atlas.dtype})'
        )
        return

    raise NotImplementedError('Population atlas: Week 3-4')


if __name__ == '__main__':
    main()
