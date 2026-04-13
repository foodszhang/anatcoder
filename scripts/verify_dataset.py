"""Verify processed and projection dataset integrity for AnatCoder experiments.

Usage:
    python scripts/verify_dataset.py \
        --processed_dir data/processed \
        --projections_dir data/projections \
        --cases case001 case002 ... case010 \
        --required_views 10 20 50
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.eval.global_metrics import compute_psnr  # noqa: E402

console = Console()

EXPECTED_SEG_LABELS = {
    '0': 'background_air',
    '1': 'bone',
    '2': 'lung',
    '3': 'liver',
    '4': 'spleen',
    '5': 'kidney',
    '6': 'pancreas',
    '7': 'heart_and_vessels',
    '8': 'gastrointestinal',
    '9': 'soft_tissue',
}


@dataclass
class CaseReport:
    """Verification summary for one case."""

    case_name: str
    nonzero_classes: int
    labels: list[int]
    fdk_psnr_50: float
    errors: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file as dict."""
    if not path.exists():
        raise FileNotFoundError(f'JSON file not found: {path}')
    content = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(content, dict):
        raise ValueError(f'JSON root must be object: {path}')
    return content


def _discover_cases(processed_dir: Path) -> list[str]:
    """Discover case directories from processed root."""
    if not processed_dir.exists():
        raise FileNotFoundError(f'Processed dir not found: {processed_dir}')
    case_names = sorted([path.name for path in processed_dir.iterdir() if path.is_dir()])
    if not case_names:
        raise FileNotFoundError(f'No cases found under: {processed_dir}')
    return case_names


def verify_case(
    case_name: str,
    processed_dir: Path,
    projections_dir: Path,
    required_views: list[int],
    expected_shape: tuple[int, int, int],
    detector_shape: tuple[int, int],
    allowed_labels: set[int],
    expected_seg_labels: dict[str, str],
    min_nonzero_classes: int,
    fdk_psnr_threshold: float,
) -> CaseReport:
    """Verify one case and return a structured report."""
    errors: list[str] = []

    case_processed = processed_dir / case_name
    volume_path = case_processed / 'volume.npy'
    seg_path = case_processed / 'seg.npy'
    seg_info_path = case_processed / 'seg_info.json'

    if not volume_path.exists():
        errors.append(f'missing volume.npy ({volume_path})')
        return CaseReport(case_name, nonzero_classes=0, labels=[], fdk_psnr_50=float('nan'), errors=errors)
    if not seg_path.exists():
        errors.append(f'missing seg.npy ({seg_path})')
        return CaseReport(case_name, nonzero_classes=0, labels=[], fdk_psnr_50=float('nan'), errors=errors)
    if not seg_info_path.exists():
        errors.append(f'missing seg_info.json ({seg_info_path})')
        return CaseReport(case_name, nonzero_classes=0, labels=[], fdk_psnr_50=float('nan'), errors=errors)

    volume = np.asarray(np.load(volume_path), dtype=np.float32)
    seg = np.asarray(np.load(seg_path), dtype=np.int16)

    if tuple(volume.shape) != expected_shape:
        errors.append(f'volume shape {volume.shape} != {expected_shape}')
    if tuple(seg.shape) != expected_shape:
        errors.append(f'seg shape {seg.shape} != {expected_shape}')

    vol_min = float(volume.min())
    vol_max = float(volume.max())
    if vol_min < -1e-6 or vol_max > 1.0 + 1e-6:
        errors.append(f'volume range out of [0,1]: min={vol_min:.6f}, max={vol_max:.6f}')

    labels = sorted(int(v) for v in np.unique(seg))
    if any(label not in allowed_labels for label in labels):
        errors.append(f'seg labels {labels} not subset of {sorted(allowed_labels)}')

    nonzero_classes = len([label for label in labels if label != 0])
    if nonzero_classes < min_nonzero_classes:
        errors.append(
            f'nonzero classes {nonzero_classes} < required {min_nonzero_classes} '
            f'(labels={labels})'
        )

    seg_info = _load_json(seg_info_path)
    seg_info_labels = seg_info.get('labels')
    if seg_info_labels != expected_seg_labels:
        errors.append('seg_info.json labels mapping does not match expected 10-class definition')

    fdk_psnr_50 = float('nan')
    case_proj_root = projections_dir / case_name
    for n_views in required_views:
        view_dir = case_proj_root / f'{n_views}views'
        projections_path = view_dir / 'projections.npy'
        angles_path = view_dir / 'angles.npy'
        fdk_path = view_dir / 'fdk_recon.npy'

        if not projections_path.exists():
            errors.append(f'missing projections.npy for {n_views} views ({projections_path})')
            continue
        if not angles_path.exists():
            errors.append(f'missing angles.npy for {n_views} views ({angles_path})')
            continue
        if not fdk_path.exists():
            errors.append(f'missing fdk_recon.npy for {n_views} views ({fdk_path})')
            continue

        projections = np.asarray(np.load(projections_path), dtype=np.float32)
        angles = np.asarray(np.load(angles_path), dtype=np.float32)
        fdk_recon = np.asarray(np.load(fdk_path), dtype=np.float32)

        expected_proj_shape = (n_views, detector_shape[0], detector_shape[1])
        if tuple(projections.shape) != expected_proj_shape:
            errors.append(f'{n_views} views projections shape {projections.shape} != {expected_proj_shape}')
        if tuple(angles.shape) != (n_views,):
            errors.append(f'{n_views} views angles shape {angles.shape} != ({n_views},)')
        if tuple(fdk_recon.shape) != expected_shape:
            errors.append(f'{n_views} views fdk_recon shape {fdk_recon.shape} != {expected_shape}')

        if n_views == 50:
            fdk_psnr_50 = compute_psnr(fdk_recon, volume, data_range=1.0)
            if np.isnan(fdk_psnr_50) or (
                np.isfinite(fdk_psnr_50) and fdk_psnr_50 <= fdk_psnr_threshold
            ):
                errors.append(f'50-view FDK PSNR {fdk_psnr_50:.4f} <= threshold {fdk_psnr_threshold:.4f}')

    return CaseReport(
        case_name=case_name,
        nonzero_classes=nonzero_classes,
        labels=labels,
        fdk_psnr_50=fdk_psnr_50,
        errors=errors,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Verify processed/projection dataset integrity.')
    parser.add_argument('--processed_dir', required=True, help='Root dir with caseXXX/volume.npy + seg.npy.')
    parser.add_argument('--projections_dir', required=True, help='Root dir with caseXXX/{K}views/*.npy.')
    parser.add_argument('--cases', nargs='*', default=None, help='Case names to verify (default: auto-discover).')
    parser.add_argument('--expected_case_count', type=int, default=10, help='Expected number of cases.')
    parser.add_argument('--required_views', nargs='+', type=int, default=[10, 20, 50], help='Required view counts.')
    parser.add_argument('--volume_size', type=int, default=128, help='Expected cubic volume size.')
    parser.add_argument('--detector_rows', type=int, default=256, help='Expected detector rows.')
    parser.add_argument('--detector_cols', type=int, default=256, help='Expected detector cols.')
    parser.add_argument('--min_nonzero_classes', type=int, default=5, help='Minimum nonzero seg classes per case.')
    parser.add_argument('--fdk_psnr_threshold', type=float, default=25.0, help='Minimum PSNR for 50-view FDK.')
    parser.add_argument('--report_json', default=None, help='Optional path to save full report JSON.')
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    projections_dir = Path(args.projections_dir)

    cases = args.cases if args.cases else _discover_cases(processed_dir)
    if int(args.expected_case_count) > 0 and len(cases) != int(args.expected_case_count):
        raise RuntimeError(
            f'Expected {args.expected_case_count} cases but got {len(cases)}: {cases}'
        )

    reports: list[CaseReport] = []
    for case_name in cases:
        report = verify_case(
            case_name=case_name,
            processed_dir=processed_dir,
            projections_dir=projections_dir,
            required_views=[int(v) for v in args.required_views],
            expected_shape=(int(args.volume_size), int(args.volume_size), int(args.volume_size)),
            detector_shape=(int(args.detector_rows), int(args.detector_cols)),
            allowed_labels=set(range(10)),
            expected_seg_labels=EXPECTED_SEG_LABELS,
            min_nonzero_classes=int(args.min_nonzero_classes),
            fdk_psnr_threshold=float(args.fdk_psnr_threshold),
        )
        reports.append(report)

    table = Table(title='Dataset Verification Summary')
    table.add_column('Case')
    table.add_column('Labels')
    table.add_column('Nonzero Classes')
    table.add_column('FDK PSNR (50-view)')
    table.add_column('Status')

    any_error = False
    for report in reports:
        status = '[green]OK[/green]' if not report.errors else '[red]FAIL[/red]'
        table.add_row(
            report.case_name,
            str(report.labels),
            str(report.nonzero_classes),
            f'{report.fdk_psnr_50:.4f}' if np.isfinite(report.fdk_psnr_50) else 'nan',
            status,
        )
        if report.errors:
            any_error = True
    console.print(table)

    if args.report_json:
        payload = {
            'cases': [
                {
                    'case_name': report.case_name,
                    'labels': report.labels,
                    'nonzero_classes': report.nonzero_classes,
                    'fdk_psnr_50': report.fdk_psnr_50,
                    'errors': report.errors,
                }
                for report in reports
            ],
        }
        Path(args.report_json).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if any_error:
        details = '\n'.join(
            f"- {report.case_name}: {'; '.join(report.errors)}"
            for report in reports
            if report.errors
        )
        raise RuntimeError(f'Dataset verification failed:\n{details}')

    console.print('[green]All dataset checks passed.[/green]')


if __name__ == '__main__':
    main()
