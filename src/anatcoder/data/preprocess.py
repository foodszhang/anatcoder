"""Preprocessing utilities for CT volumes and segmentation masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from anatcoder.utils.io import save_numpy

console = Console()


def _compute_resampled_size(image: sitk.Image, target_spacing: tuple[float, float, float]) -> list[int]:
    """Compute output size when resampling to a new spacing."""
    old_size = np.asarray(list(image.GetSize()), dtype=np.float64)
    old_spacing = np.asarray(list(image.GetSpacing()), dtype=np.float64)
    new_spacing = np.asarray(list(target_spacing), dtype=np.float64)
    new_size = np.round(old_size * (old_spacing / new_spacing)).astype(np.int32)
    return [max(int(v), 1) for v in new_size.tolist()]


def _resample_image(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    interpolator: int,
) -> sitk.Image:
    """Resample a SimpleITK image to target spacing."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(tuple(float(v) for v in target_spacing))
    resampler.SetSize(_compute_resampled_size(image, target_spacing))
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _resample_to_reference(image: sitk.Image, reference_image: sitk.Image, interpolator: int) -> sitk.Image:
    """Resample an image to reference image grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _center_crop_or_pad(volume: np.ndarray, crop_size: int, pad_value: float) -> np.ndarray:
    """Center-crop or zero-pad a 3D volume to ``crop_size^3``."""
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3D array, got shape: {volume.shape}')

    target_shape = np.asarray([crop_size, crop_size, crop_size], dtype=np.int32)
    src_shape = np.asarray(volume.shape, dtype=np.int32)

    # Crop.
    start = np.maximum((src_shape - target_shape) // 2, 0)
    end = np.minimum(start + target_shape, src_shape)
    cropped = volume[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

    # Pad.
    dst = np.full(tuple(target_shape.tolist()), pad_value, dtype=volume.dtype)
    c_shape = np.asarray(cropped.shape, dtype=np.int32)
    d_start = np.maximum((target_shape - c_shape) // 2, 0)
    d_end = d_start + c_shape
    dst[d_start[0] : d_end[0], d_start[1] : d_end[1], d_start[2] : d_end[2]] = cropped
    return dst


def _case_name_from_ct_path(ct_path: Path) -> str:
    """Infer output case name from a CT file path."""
    if ct_path.parent.name in {'imagesTr', 'imagesTs'}:
        return ct_path.name.replace('.nii.gz', '').replace('.nii', '')
    return ct_path.parent.name


def _detect_seg_source(ct_path: Path) -> Path | None:
    """Auto-detect segmentation source for TotalSegmentator/AMOS layouts."""
    parent = ct_path.parent

    direct_candidates = [
        parent / 'segmentation.nii.gz',
        parent / 'seg.nii.gz',
        parent / 'label.nii.gz',
        parent / 'labels.nii.gz',
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    for directory_name in ('segmentations', 'labels'):
        directory = parent / directory_name
        if directory.is_dir():
            return directory

    if parent.name == 'imagesTr':
        labels_dir = parent.parent / 'labelsTr'
        same_name = labels_dir / ct_path.name
        if same_name.exists():
            return same_name
        maybe_amos = labels_dir / ct_path.name.replace('_0000.nii.gz', '.nii.gz')
        if maybe_amos.exists():
            return maybe_amos

    return None


def _merge_multiorgan_segmentations(seg_dir: Path, reference_image: sitk.Image) -> np.ndarray:
    """Merge many binary organ segmentations into one multi-class label map."""
    organ_files = sorted(seg_dir.glob('*.nii.gz')) + sorted(seg_dir.glob('*.nii'))
    if not organ_files:
        raise FileNotFoundError(f'No organ segmentation files found in: {seg_dir}')

    merged = np.zeros(sitk.GetArrayFromImage(reference_image).shape, dtype=np.int16)
    for class_idx, organ_file in enumerate(organ_files, start=1):
        organ_image = sitk.ReadImage(str(organ_file))
        resampled = _resample_to_reference(organ_image, reference_image, sitk.sitkNearestNeighbor)
        organ_mask = sitk.GetArrayFromImage(resampled)
        merged[organ_mask > 0] = class_idx
    return merged


def preprocess_segmentation(
    seg_path: str,
    reference_image: sitk.Image,
    crop_size: int = 128,
) -> np.ndarray:
    """Preprocess segmentation labels and align them with preprocessed CT.

    Args:
        seg_path: Path to a segmentation file or directory.
        reference_image: Reference image used for resampling alignment.
        crop_size: Output cube side length.

    Returns:
        Preprocessed segmentation as ``[D, H, W]`` integer array.
    """
    seg_source = Path(seg_path)
    if not seg_source.exists():
        raise FileNotFoundError(f'Segmentation source not found: {seg_source}')

    if seg_source.is_dir():
        seg_array = _merge_multiorgan_segmentations(seg_source, reference_image)
    else:
        seg_image = sitk.ReadImage(str(seg_source))
        seg_resampled = _resample_to_reference(seg_image, reference_image, sitk.sitkNearestNeighbor)
        seg_array = sitk.GetArrayFromImage(seg_resampled)

    seg_array = _center_crop_or_pad(seg_array.astype(np.int16, copy=False), crop_size=crop_size, pad_value=0)
    return seg_array.astype(np.int16, copy=False)


def preprocess_ct(
    input_path: str,
    output_dir: str,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_window: tuple[float, float] = (-160.0, 240.0),
    crop_size: int = 128,
) -> dict:
    """Preprocess one CT volume.

    Steps:
        1. Load NIfTI with SimpleITK.
        2. Resample to ``target_spacing`` using linear interpolation.
        3. Clip HU values to ``hu_window`` and normalize to ``[0, 1]``.
        4. Center-crop or zero-pad to ``crop_size^3``.
        5. Save ``volume.npy``.
        6. If segmentation exists, preprocess with nearest-neighbor interpolation and save ``seg.npy``.

    Returns:
        Dictionary with output paths and metadata.
    """
    ct_path = Path(input_path)
    if not ct_path.exists():
        raise FileNotFoundError(f'CT file not found: {ct_path}')

    case_name = _case_name_from_ct_path(ct_path)
    case_out_dir = Path(output_dir) / case_name
    case_out_dir.mkdir(parents=True, exist_ok=True)

    console.log(f'[cyan]Preprocessing case[/cyan] {case_name} from {ct_path}')

    ct_image = sitk.ReadImage(str(ct_path))
    ct_resampled = _resample_image(ct_image, target_spacing=target_spacing, interpolator=sitk.sitkLinear)
    ct_array = sitk.GetArrayFromImage(ct_resampled).astype(np.float32, copy=False)

    hu_min, hu_max = float(hu_window[0]), float(hu_window[1])
    if hu_max <= hu_min:
        raise ValueError(f'Invalid HU window: {hu_window}')

    ct_array = np.clip(ct_array, hu_min, hu_max)
    ct_array = (ct_array - hu_min) / (hu_max - hu_min)
    ct_array = _center_crop_or_pad(ct_array.astype(np.float32, copy=False), crop_size=crop_size, pad_value=0.0)

    volume_path = case_out_dir / 'volume.npy'
    save_numpy(ct_array.astype(np.float32, copy=False), volume_path)

    seg_source = _detect_seg_source(ct_path)
    seg_out_path: Path | None = None
    if seg_source is not None:
        seg_array = preprocess_segmentation(str(seg_source), reference_image=ct_resampled, crop_size=crop_size)
        seg_out_path = case_out_dir / 'seg.npy'
        save_numpy(seg_array, seg_out_path)

    metadata = {
        'case_name': case_name,
        'input_path': str(ct_path),
        'seg_source': str(seg_source) if seg_source is not None else None,
        'output_dir': str(case_out_dir),
        'volume_path': str(volume_path),
        'seg_path': str(seg_out_path) if seg_out_path is not None else None,
        'target_spacing': [float(v) for v in target_spacing],
        'hu_window': [hu_min, hu_max],
        'crop_size': int(crop_size),
        'original_spacing': [float(v) for v in ct_image.GetSpacing()],
        'original_size': [int(v) for v in ct_image.GetSize()],
        'resampled_size': [int(v) for v in ct_resampled.GetSize()],
        'output_shape_zyx': [int(v) for v in ct_array.shape],
    }

    metadata_path = case_out_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def _is_nifti(path: Path) -> bool:
    """Return true for NIfTI file names."""
    return path.name.endswith('.nii') or path.name.endswith('.nii.gz')


def _discover_ct_files(input_dir: Path) -> list[Path]:
    """Discover CT files from mixed dataset layouts."""
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    ct_files: list[Path] = []

    amos_images = input_dir / 'imagesTr'
    if amos_images.is_dir():
        ct_files.extend(sorted([p for p in amos_images.iterdir() if p.is_file() and _is_nifti(p)]))

    for case_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        if case_dir.name in {'imagesTr', 'labelsTr'}:
            continue

        priority = [
            case_dir / 'ct.nii.gz',
            case_dir / 'ct.nii',
            case_dir / 'image.nii.gz',
            case_dir / 'image.nii',
        ]
        selected: Path | None = None
        for candidate in priority:
            if candidate.exists():
                selected = candidate
                break

        if selected is None:
            nii_files = sorted(case_dir.glob('*.nii.gz')) + sorted(case_dir.glob('*.nii'))
            for maybe_ct in nii_files:
                lname = maybe_ct.name.lower()
                if any(tag in lname for tag in ('seg', 'label', 'mask')):
                    continue
                selected = maybe_ct
                break

        if selected is not None:
            ct_files.append(selected)

    for file_path in sorted([p for p in input_dir.iterdir() if p.is_file() and _is_nifti(p)]):
        if file_path not in ct_files:
            ct_files.append(file_path)

    deduped = sorted(set(ct_files))
    return deduped


def batch_preprocess(
    input_dir: str,
    output_dir: str,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_window: tuple[float, float] = (-160.0, 240.0),
    crop_size: int = 128,
) -> list[dict]:
    """Batch preprocess all CT files under ``input_dir``.

    This function supports both:
    - TotalSegmentator-like case directories with ``ct.nii.gz`` + ``segmentations/``.
    - AMOS-like ``imagesTr/`` + ``labelsTr/`` layout.

    Rich progress bars are used for CLI feedback.

    Returns:
        A list of per-case metadata dictionaries.
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    ct_files = _discover_ct_files(input_root)
    if not ct_files:
        raise FileNotFoundError(f'No CT files found under: {input_root}')

    console.print(f'[bold]Found {len(ct_files)} CT file(s) for preprocessing.[/bold]')

    results: list[dict] = []
    failures: list[tuple[Path, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task('Preprocessing CT volumes', total=len(ct_files))
        for ct_path in ct_files:
            try:
                result = preprocess_ct(
                    input_path=str(ct_path),
                    output_dir=str(output_root),
                    target_spacing=target_spacing,
                    hu_window=hu_window,
                    crop_size=crop_size,
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                failures.append((ct_path, str(exc)))
                console.print(f'[red]Failed[/red] {ct_path}: {exc}')
            finally:
                progress.advance(task_id)

    if failures:
        details = "\n".join(f"- {path}: {msg}" for path, msg in failures)
        raise RuntimeError(f"Preprocessing failed for {len(failures)} case(s):\n{details}")

    console.print(f'[green]Preprocessing finished successfully for {len(results)} case(s).[/green]')
    return results
