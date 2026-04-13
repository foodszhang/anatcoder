"""Preprocess TotalSegmentator v2 CT + masks into 128^3 numpy cases.

Usage:
    python scripts/preprocess_totalseg.py \
        --input_dir data/raw/totalsegmentator \
        --output_dir data/processed \
        --num_cases 10 \
        --crop_size 128 \
        --spacing 1.0 1.0 1.0 \
        --hu_window -160 240
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.utils.io import save_numpy  # noqa: E402

console = Console()

SEG_CLASS_NAMES: dict[int, str] = {
    0: 'background_air',
    1: 'bone',
    2: 'lung',
    3: 'liver',
    4: 'spleen',
    5: 'kidney',
    6: 'pancreas',
    7: 'heart_and_vessels',
    8: 'gastrointestinal',
    9: 'soft_tissue',
}


@dataclass(frozen=True)
class TotalSegCase:
    """Pointer to one TotalSegmentator case."""

    case_id: str
    ct_path: Path
    seg_dir: Path


def _strip_nii_suffix(name: str) -> str:
    """Strip .nii or .nii.gz from filename."""
    if name.endswith('.nii.gz'):
        return name[: -len('.nii.gz')]
    if name.endswith('.nii'):
        return name[: -len('.nii')]
    return name


def _find_ct_file(case_dir: Path) -> Path | None:
    """Find CT file in one case directory."""
    candidates = [
        case_dir / 'ct.nii.gz',
        case_dir / 'ct.nii',
        case_dir / 'image.nii.gz',
        case_dir / 'image.nii',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_image(path: Path) -> sitk.Image:
    """Read NIfTI with SimpleITK and fallback to nibabel for malformed direction cosines."""
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError as exc:
        if 'orthonormal direction cosines' not in str(exc).lower():
            raise

    nifti = nib.load(str(path))
    array_xyz = np.asarray(nifti.dataobj)
    if array_xyz.ndim != 3:
        raise ValueError(f'Expected 3D image, got shape={array_xyz.shape} ({path})')

    if np.issubdtype(array_xyz.dtype, np.floating):
        array_xyz = array_xyz.astype(np.float32, copy=False)
    else:
        array_xyz = array_xyz.astype(np.int16, copy=False)

    array_zyx = np.transpose(array_xyz, (2, 1, 0))
    image = sitk.GetImageFromArray(array_zyx)
    zooms = nifti.header.get_zooms()[:3]
    image.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _compute_resampled_size(image: sitk.Image, target_spacing: tuple[float, float, float]) -> list[int]:
    """Compute output size when changing spacing."""
    old_size = np.asarray(list(image.GetSize()), dtype=np.float64)
    old_spacing = np.asarray(list(image.GetSpacing()), dtype=np.float64)
    new_spacing = np.asarray(list(target_spacing), dtype=np.float64)
    new_size = np.round(old_size * (old_spacing / new_spacing)).astype(np.int32)
    return [max(int(v), 1) for v in new_size.tolist()]


def _resample_image(image: sitk.Image, spacing: tuple[float, float, float], interpolator: int) -> sitk.Image:
    """Resample image to target spacing."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(tuple(float(v) for v in spacing))
    resampler.SetSize(_compute_resampled_size(image, spacing))
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _resample_to_reference(image: sitk.Image, reference_image: sitk.Image, interpolator: int) -> sitk.Image:
    """Resample image to reference grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _center_crop_or_pad(volume: np.ndarray, crop_size: int, pad_value: float) -> np.ndarray:
    """Center-crop or pad to crop_size^3."""
    if volume.ndim != 3:
        raise ValueError(f'Expected 3D array, got shape={volume.shape}')

    target_shape = np.asarray([crop_size, crop_size, crop_size], dtype=np.int32)
    src_shape = np.asarray(volume.shape, dtype=np.int32)

    src_start = np.maximum((src_shape - target_shape) // 2, 0)
    src_end = np.minimum(src_start + target_shape, src_shape)
    cropped = volume[src_start[0] : src_end[0], src_start[1] : src_end[1], src_start[2] : src_end[2]]

    dst = np.full(tuple(target_shape.tolist()), pad_value, dtype=volume.dtype)
    crop_shape = np.asarray(cropped.shape, dtype=np.int32)
    dst_start = np.maximum((target_shape - crop_shape) // 2, 0)
    dst_end = dst_start + crop_shape
    dst[dst_start[0] : dst_end[0], dst_start[1] : dst_end[1], dst_start[2] : dst_end[2]] = cropped
    return dst


def _map_label_to_class(label_name: str) -> int | None:
    """Map TotalSegmentator organ label name to 10-class ID."""
    name = label_name.lower()

    if name.startswith(
        (
            'vertebrae_',
            'rib_',
            'scapula_',
            'clavicula_',
            'clavicle_',
            'hip_',
            'femur_',
        )
    ) or name == 'sacrum':
        return 1

    if name.startswith('lung_'):
        return 2

    if name == 'liver':
        return 3

    if name == 'spleen':
        return 4

    if name in {'kidney_left', 'kidney_right'}:
        return 5

    if name == 'pancreas':
        return 6

    if name in {'heart', 'aorta', 'pulmonary_artery', 'inferior_vena_cava'}:
        return 7

    if name in {'stomach', 'small_bowel', 'duodenum'} or name.startswith('colon'):
        return 8

    if name in {'subcutaneous_fat', 'torso_fat'} or 'muscle' in name:
        return 9

    return None


def _discover_totalseg_cases(input_dir: Path) -> list[TotalSegCase]:
    """Discover TotalSegmentator case dirs with ct + segmentations."""
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    cases: list[TotalSegCase] = []
    for case_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        seg_dir = case_dir / 'segmentations'
        if not seg_dir.is_dir():
            continue
        ct_path = _find_ct_file(case_dir)
        if ct_path is None:
            continue
        cases.append(TotalSegCase(case_id=case_dir.name, ct_path=ct_path, seg_dir=seg_dir))
    return cases


def _estimate_case_scale(case: TotalSegCase) -> float:
    """Estimate body scale by physical image volume for diversity sampling."""
    image = _read_image(case.ct_path)
    size = np.asarray(list(image.GetSize()), dtype=np.float64)
    spacing = np.asarray(list(image.GetSpacing()), dtype=np.float64)
    return float(np.prod(size * spacing))


def _case_has_chest_abdomen(case: TotalSegCase) -> bool:
    """Heuristic filter: case should include both thorax and abdomen organs."""
    labels = {
        _strip_nii_suffix(path.name)
        for path in sorted(list(case.seg_dir.glob('*.nii.gz')) + list(case.seg_dir.glob('*.nii')))
    }
    has_lung = any(name.startswith('lung_') for name in labels)
    has_liver = 'liver' in labels
    has_kidney = 'kidney_left' in labels or 'kidney_right' in labels
    has_heart_or_vessel = bool({'heart', 'aorta', 'pulmonary_artery'} & labels)
    return has_lung and has_liver and has_kidney and has_heart_or_vessel


def _pick_diverse_cases(cases: list[TotalSegCase], n_cases: int) -> list[TotalSegCase]:
    """Pick n cases by physical-size stratification."""
    if n_cases <= 0:
        raise ValueError(f'n_cases must be positive, got: {n_cases}')

    chest_abd = [case for case in cases if _case_has_chest_abdomen(case)]
    pool = chest_abd if len(chest_abd) >= n_cases else cases
    if len(pool) < n_cases:
        raise RuntimeError(
            f'Not enough cases to select {n_cases}. Found={len(pool)}, '
            f'valid chest-abd={len(chest_abd)}, total discovered={len(cases)}.'
        )

    scored = sorted([(case, _estimate_case_scale(case)) for case in pool], key=lambda item: item[1])
    if len(scored) == n_cases:
        return [item[0] for item in scored]

    indices = np.linspace(0, len(scored) - 1, n_cases, dtype=np.float64)
    selected_indices = sorted({int(round(v)) for v in indices})
    while len(selected_indices) < n_cases:
        for idx in range(len(scored)):
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) == n_cases:
                    break
    selected_indices = sorted(selected_indices[:n_cases])
    return [scored[idx][0] for idx in selected_indices]


def _select_cases(cases: list[TotalSegCase], case_ids: list[str] | None, n_cases: int) -> list[TotalSegCase]:
    """Select requested cases by IDs or by auto-picking."""
    if case_ids:
        selected: list[TotalSegCase] = []
        by_id = {case.case_id: case for case in cases}
        for case_id in case_ids:
            if case_id not in by_id:
                raise KeyError(f'Requested case not found: {case_id}')
            selected.append(by_id[case_id])
        return selected
    return _pick_diverse_cases(cases, n_cases=n_cases)


def _merge_segmentations(
    seg_dir: Path,
    reference_image: sitk.Image,
) -> tuple[np.ndarray, dict[int, set[str]], set[str]]:
    """Merge 117 masks into one 10-class volume on reference grid."""
    organ_files = sorted(list(seg_dir.glob('*.nii.gz')) + list(seg_dir.glob('*.nii')))
    if not organ_files:
        raise FileNotFoundError(f'No segmentation masks found under: {seg_dir}')

    merged = np.zeros(sitk.GetArrayFromImage(reference_image).shape, dtype=np.int16)
    fallback_nonzero = np.zeros_like(merged, dtype=bool)
    class_sources: dict[int, set[str]] = {class_id: set() for class_id in SEG_CLASS_NAMES}
    fallback_labels: set[str] = set()

    for organ_path in organ_files:
        label_name = _strip_nii_suffix(organ_path.name)
        class_id = _map_label_to_class(label_name)

        organ_image = _read_image(organ_path)
        organ_resampled = _resample_to_reference(organ_image, reference_image, sitk.sitkNearestNeighbor)
        organ_mask = sitk.GetArrayFromImage(organ_resampled) > 0
        if not np.any(organ_mask):
            continue

        if class_id is None:
            fallback_nonzero |= organ_mask
            fallback_labels.add(label_name)
            continue

        assign = organ_mask & ((merged == 0) | (merged == class_id))
        merged[assign] = np.int16(class_id)
        class_sources[class_id].add(label_name)

    fallback_assign = (merged == 0) & fallback_nonzero
    if np.any(fallback_assign):
        merged[fallback_assign] = np.int16(9)
        class_sources[9].update(fallback_labels)

    return merged, class_sources, fallback_labels


def preprocess_case(
    case: TotalSegCase,
    output_case_dir: Path,
    spacing: tuple[float, float, float],
    hu_window: tuple[float, float],
    crop_size: int,
) -> dict[str, Any]:
    """Preprocess one TotalSegmentator case."""
    if hu_window[1] <= hu_window[0]:
        raise ValueError(f'Invalid HU window: {hu_window}')

    output_case_dir.mkdir(parents=True, exist_ok=True)

    ct_image = _read_image(case.ct_path)
    ct_resampled = _resample_image(ct_image, spacing=spacing, interpolator=sitk.sitkLinear)
    ct_array = sitk.GetArrayFromImage(ct_resampled).astype(np.float32, copy=False)

    hu_min, hu_max = float(hu_window[0]), float(hu_window[1])
    ct_array = np.clip(ct_array, hu_min, hu_max)
    ct_array = (ct_array - hu_min) / (hu_max - hu_min)
    ct_array = _center_crop_or_pad(ct_array.astype(np.float32, copy=False), crop_size=crop_size, pad_value=0.0)

    seg_merged, class_sources, fallback_labels = _merge_segmentations(case.seg_dir, ct_resampled)
    seg_merged = _center_crop_or_pad(seg_merged.astype(np.int16, copy=False), crop_size=crop_size, pad_value=0)

    save_numpy(ct_array.astype(np.float32, copy=False), output_case_dir / 'volume.npy')
    save_numpy(seg_merged.astype(np.int16, copy=False), output_case_dir / 'seg.npy')

    seg_info = {
        'labels': {str(k): v for k, v in SEG_CLASS_NAMES.items()},
        'class_sources': {str(k): sorted(v) for k, v in class_sources.items()},
        'fallback_to_soft_tissue': sorted(fallback_labels),
    }
    (output_case_dir / 'seg_info.json').write_text(json.dumps(seg_info, indent=2), encoding='utf-8')

    metadata = {
        'source_case_id': case.case_id,
        'source_ct_path': str(case.ct_path),
        'source_seg_dir': str(case.seg_dir),
        'target_spacing': [float(v) for v in spacing],
        'hu_window': [hu_min, hu_max],
        'crop_size': int(crop_size),
        'original_spacing': [float(v) for v in ct_image.GetSpacing()],
        'original_size': [int(v) for v in ct_image.GetSize()],
        'resampled_size': [int(v) for v in ct_resampled.GetSize()],
        'output_shape_zyx': [int(v) for v in ct_array.shape],
    }
    (output_case_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description='Preprocess TotalSegmentator v2 to 10-class 128^3 numpy cases.')
    parser.add_argument('--input_dir', required=True, help='TotalSegmentator root directory.')
    parser.add_argument('--output_dir', required=True, help='Output processed directory.')
    parser.add_argument('--num_cases', type=int, default=10, help='Number of cases to preprocess when auto-selecting.')
    parser.add_argument('--case_ids', nargs='*', default=None, help='Optional explicit source case IDs.')
    parser.add_argument(
        '--spacing',
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=('SZ', 'SY', 'SX'),
        help='Target spacing in mm.',
    )
    parser.add_argument('--crop_size', type=int, default=128, help='Output cubic volume size.')
    parser.add_argument(
        '--hu_window',
        nargs=2,
        type=float,
        default=(-160.0, 240.0),
        metavar=('HU_MIN', 'HU_MAX'),
        help='HU window for normalization.',
    )
    parser.add_argument(
        '--output_case_prefix',
        default='case',
        help='Output case prefix, default writes case001..case010.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing output case directories.',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_totalseg_cases(input_dir)
    if not discovered:
        raise FileNotFoundError(
            f'No TotalSegmentator-style cases found in {input_dir}. '
            'Expected per-case dirs containing ct.nii.gz and segmentations/*.nii.gz.'
        )

    selected = _select_cases(discovered, case_ids=args.case_ids, n_cases=int(args.num_cases))
    console.print(
        f'[bold cyan]Preprocessing {len(selected)} case(s)[/bold cyan] '
        f'from {len(discovered)} discovered under {input_dir}'
    )

    summary_rows: list[tuple[str, str, str]] = []
    for idx, case in enumerate(selected, start=1):
        output_case_name = f'{args.output_case_prefix}{idx:03d}'
        output_case_dir = output_dir / output_case_name
        if output_case_dir.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists: {output_case_dir} (use --overwrite to replace)')

        metadata = preprocess_case(
            case=case,
            output_case_dir=output_case_dir,
            spacing=tuple(float(v) for v in args.spacing),
            hu_window=(float(args.hu_window[0]), float(args.hu_window[1])),
            crop_size=int(args.crop_size),
        )
        summary_rows.append(
            (
                output_case_name,
                case.case_id,
                f"{metadata['output_shape_zyx'][0]}^3",
            )
        )

    summary = Table(title='TotalSegmentator Preprocess Summary')
    summary.add_column('Output Case')
    summary.add_column('Source Case')
    summary.add_column('Shape')
    for row in summary_rows:
        summary.add_row(*row)
    console.print(summary)
    console.print('[green]Done.[/green] volume.npy, seg.npy, seg_info.json saved for all cases.')


if __name__ == '__main__':
    main()
