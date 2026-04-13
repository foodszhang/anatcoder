"""Tests for Week-3 dataset prep scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.preprocess_totalseg import _map_label_to_class  # noqa: E402
from scripts.verify_dataset import EXPECTED_SEG_LABELS, verify_case  # noqa: E402


@pytest.mark.parametrize(
    ('label_name', 'expected_class'),
    [
        ('vertebrae_L3', 1),
        ('rib_left_4', 1),
        ('scapula_right', 1),
        ('lung_upper_lobe_left', 2),
        ('liver', 3),
        ('spleen', 4),
        ('kidney_right', 5),
        ('pancreas', 6),
        ('aorta', 7),
        ('small_bowel', 8),
        ('torso_fat', 9),
        ('random_unknown_label', None),
    ],
)
def test_map_label_to_class(label_name: str, expected_class: int | None) -> None:
    """Label-name merge rules should map organs into expected coarse classes."""
    assert _map_label_to_class(label_name) == expected_class


def test_verify_case_happy_path(tmp_path: Path) -> None:
    """verify_case should pass for a valid minimal synthetic case."""
    processed_dir = tmp_path / 'processed'
    projections_dir = tmp_path / 'projections'
    case_name = 'case001'

    case_processed = processed_dir / case_name
    case_processed.mkdir(parents=True, exist_ok=True)

    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 1.0
    seg = np.zeros((8, 8, 8), dtype=np.int16)
    seg[:2, :, :] = 1
    seg[2:3, :, :] = 2
    seg[3:4, :, :] = 3
    seg[4:5, :, :] = 4
    seg[5:6, :, :] = 5

    np.save(case_processed / 'volume.npy', volume)
    np.save(case_processed / 'seg.npy', seg)
    (case_processed / 'seg_info.json').write_text(
        json.dumps({'labels': EXPECTED_SEG_LABELS}, indent=2),
        encoding='utf-8',
    )

    for n_views in (10, 20, 50):
        view_dir = projections_dir / case_name / f'{n_views}views'
        view_dir.mkdir(parents=True, exist_ok=True)
        np.save(view_dir / 'projections.npy', np.zeros((n_views, 8, 8), dtype=np.float32))
        np.save(view_dir / 'angles.npy', np.linspace(0.0, 1.0, n_views, dtype=np.float32))
        np.save(view_dir / 'fdk_recon.npy', volume.astype(np.float32))

    report = verify_case(
        case_name=case_name,
        processed_dir=processed_dir,
        projections_dir=projections_dir,
        required_views=[10, 20, 50],
        expected_shape=(8, 8, 8),
        detector_shape=(8, 8),
        allowed_labels=set(range(10)),
        expected_seg_labels=EXPECTED_SEG_LABELS,
        min_nonzero_classes=5,
        fdk_psnr_threshold=25.0,
    )
    assert report.errors == []
    assert report.nonzero_classes == 5
    assert report.labels == [0, 1, 2, 3, 4, 5]
