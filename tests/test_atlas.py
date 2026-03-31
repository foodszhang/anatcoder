"""Tests for Week 1 Oracle atlas pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.atlas import AtlasBuilder, AtlasQuerier


def test_oracle_atlas_shape(tmp_path: Path) -> None:
    """测试 Oracle Atlas 输出 shape。"""
    seg = np.random.default_rng(0).integers(0, 5, size=(32, 32, 32), dtype=np.int16)
    seg_path = tmp_path / 'seg.npy'
    np.save(seg_path, seg)

    atlas = AtlasBuilder.from_oracle(str(seg_path), n_classes=5)
    assert atlas.shape == (5, 32, 32, 32)
    assert atlas.dtype == np.float32


def test_oracle_atlas_is_onehot(tmp_path: Path) -> None:
    """验证 Oracle Atlas 是 one-hot。"""
    seg = np.random.default_rng(1).integers(0, 5, size=(16, 16, 16), dtype=np.int16)
    seg_path = tmp_path / 'seg.npy'
    np.save(seg_path, seg)

    atlas = AtlasBuilder.from_oracle(str(seg_path), n_classes=5)
    summed = atlas.sum(axis=0)
    assert np.allclose(summed, 1.0)
    assert np.all((atlas == 0.0) | (atlas == 1.0))


def test_atlas_querier_shape(tmp_path: Path) -> None:
    """测试 AtlasQuerier 查询输出 shape。"""
    seg = np.random.default_rng(2).integers(0, 4, size=(20, 18, 16), dtype=np.int16)
    seg_path = tmp_path / 'seg.npy'
    atlas_path = tmp_path / 'atlas.npy'
    np.save(seg_path, seg)
    np.save(atlas_path, AtlasBuilder.from_oracle(str(seg_path), n_classes=4))

    querier = AtlasQuerier(str(atlas_path))
    coords = torch.rand((13, 3), dtype=torch.float32)
    probs = querier.query(coords)

    assert probs.shape == (13, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(13), atol=1e-4)


def test_atlas_querier_boundary(tmp_path: Path) -> None:
    """测试边界坐标（0, 1）不会越界。"""
    seg = np.zeros((8, 8, 8), dtype=np.int16)
    seg[4:, :, :] = 1
    seg_path = tmp_path / 'seg.npy'
    atlas_path = tmp_path / 'atlas.npy'
    np.save(seg_path, seg)
    np.save(atlas_path, AtlasBuilder.from_oracle(str(seg_path), n_classes=2))

    querier = AtlasQuerier(str(atlas_path))
    boundary_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    probs = querier.query(boundary_coords)

    assert probs.shape == (4, 2)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-4)
