"""Tests for dataset flattening and scaling in NAF ray mode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.dataset import CTProjectionDataset
from anatcoder.utils.geometry import CBCTGeometry


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / 'processed' / 'case001'
    proj_case_dir = tmp_path / 'projections' / 'case001' / '1views'
    case_dir.mkdir(parents=True, exist_ok=True)
    proj_case_dir.mkdir(parents=True, exist_ok=True)

    vol = np.zeros((8, 8, 8), dtype=np.float32)
    np.save(case_dir / 'volume.npy', vol)

    proj = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ],
        dtype=np.float32,
    )  # [K=1, rows=2, cols=3]
    np.save(proj_case_dir / 'projections.npy', proj)
    np.save(proj_case_dir / 'angles.npy', np.array([0.0], dtype=np.float32))
    return case_dir, proj_case_dir.parent


def test_naf_mode_uses_c_order_without_scaling(tmp_path: Path) -> None:
    case_dir, proj_dir = _write_case(tmp_path)
    geo = CBCTGeometry(n_voxel=[8, 8, 8], d_voxel=[1.0, 1.0, 1.0], n_detector=[2, 3], d_detector=[1.5, 1.5])
    ds = CTProjectionDataset(
        case_dir=str(case_dir),
        proj_dir=str(proj_dir),
        n_views=1,
        geo=geo,
        use_naf_rays=True,
    )
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(ds._gt_pixels, expected)


def test_legacy_mode_keeps_fortran_flatten(tmp_path: Path) -> None:
    case_dir, proj_dir = _write_case(tmp_path)
    geo = CBCTGeometry(n_voxel=[8, 8, 8], d_voxel=[1.0, 1.0, 1.0], n_detector=[2, 3], d_detector=[1.5, 1.5])
    ds = CTProjectionDataset(
        case_dir=str(case_dir),
        proj_dir=str(proj_dir),
        n_views=1,
        geo=geo,
        use_naf_rays=False,
    )
    expected = np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], dtype=np.float32)
    assert np.allclose(ds._gt_pixels, expected)
