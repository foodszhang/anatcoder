"""诊断射线坐标系是否与 TIGRE 一致。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anatcoder.models.ray_utils import (
    generate_rays_for_view,
    normalize_coords,
    sample_points_along_rays,
)
from anatcoder.models.renderer import VolumeRenderer
from anatcoder.utils.geometry import CBCTGeometry

# ---------- 配置 ----------
N = 128  # 和实际训练一致
geo = CBCTGeometry(
    DSD=1536.0, DSO=1000.0,
    n_voxel=[N, N, N], d_voxel=[1.0, 1.0, 1.0],
    n_detector=[256, 256], d_detector=[1.5, 1.5],
)

# ---------- 加载真实数据 ----------
volume = np.load("data/processed/case001/volume.npy").astype(np.float32)
angles = np.load("data/projections/case001/50views/angles.npy").astype(np.float32)
projections = np.load("data/projections/case001/50views/projections.npy").astype(np.float32)

print(f"Volume shape: {volume.shape}, range: [{volume.min():.4f}, {volume.max():.4f}]")
print(f"Projections shape: {projections.shape}, range: [{projections.min():.4f}, {projections.max():.4f}]")
print(f"Angles shape: {angles.shape}, first 5: {np.rad2deg(angles[:5])}")

# ---------- TIGRE 前向投影（ground truth） ----------
import tigre
tigre_geo = geo.to_tigre_geometry()
# 只用第 0 个角度做对比
angle_0 = angles[0:1]
tigre_proj_0 = tigre.Ax(volume, tigre_geo, angle_0)[0]  # [rows, cols]
stored_proj_0 = projections[0]  # [rows, cols]

# 先检查存储的投影和 TIGRE 实时投影是否一致
proj_diff = np.abs(tigre_proj_0 - stored_proj_0)
print(f"\n=== 存储投影 vs TIGRE 实时投影 ===")
print(f"Max diff: {proj_diff.max():.6f}, Mean diff: {proj_diff.mean():.6f}")
print(f"Stored range: [{stored_proj_0.min():.4f}, {stored_proj_0.max():.4f}]")
print(f"TIGRE range: [{tigre_proj_0.min():.4f}, {tigre_proj_0.max():.4f}]")

# ---------- 用 ray_utils 做 line integral ----------
origins, directions = generate_rays_for_view(geo, angle=float(angles[0]), device=torch.device("cpu"))

diag = float(np.linalg.norm(np.array(geo.n_voxel, dtype=np.float32) * np.array(geo.d_voxel, dtype=np.float32)))
near = float(geo.DSO - 0.5 * diag)
far = float(geo.DSO + 0.5 * diag)

print(f"\n=== 射线参数 ===")
print(f"Origins shape: {origins.shape}")
print(f"Origin[0]: {origins[0].numpy()}")
print(f"Direction[0]: {directions[0].numpy()}")
print(f"near={near:.2f}, far={far:.2f}")

points, step_sizes = sample_points_along_rays(
    origins, directions, n_samples=512, near=near, far=far, perturb=False
)

volume_size_mm = [float(N * 1.0)] * 3
points_norm = normalize_coords(points, volume_size_mm)

print(f"Points norm range: [{points_norm.min():.4f}, {points_norm.max():.4f}]")
print(f"Points norm center sample: {points_norm[origins.shape[0]//2, 256, :].numpy()}")

# 从 volume 中三线性插值查密度
phantom_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
# grid_sample 需要 [-1,1]，且最后一维是 (x=dim2, y=dim1, z=dim0)
grid = points_norm * 2.0 - 1.0
grid_for_sample = grid[..., [2, 1, 0]]  # 翻转轴序给 grid_sample

n_rays, n_samples = points_norm.shape[0], points_norm.shape[1]
flat_grid = grid_for_sample.reshape(1, 1, n_rays * n_samples, 1, 3)
sampled = torch.nn.functional.grid_sample(
    phantom_tensor, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True
)
densities = sampled.reshape(n_rays, n_samples, 1)

renderer = VolumeRenderer()
our_proj = renderer(densities, step_sizes).detach().numpy()
our_proj_2d = our_proj.reshape(256, 256)

# ---------- 全面对比 ----------
print(f"\n=== ray_utils 投影 vs TIGRE 投影 ===")
print(f"Our range: [{our_proj_2d.min():.4f}, {our_proj_2d.max():.4f}]")
print(f"TIGRE range: [{tigre_proj_0.min():.4f}, {tigre_proj_0.max():.4f}]")
print(f"Our mean (nonzero): {our_proj_2d[our_proj_2d > 0.001].mean():.4f}")
print(f"TIGRE mean (nonzero): {tigre_proj_0[tigre_proj_0 > 0.001].mean():.4f}")

# Pearson 相关
mask = (tigre_proj_0 > 0.001) | (our_proj_2d > 0.001)
if mask.sum() > 10:
    corr = np.corrcoef(our_proj_2d[mask], tigre_proj_0[mask])[0, 1]
    print(f"Pixel-wise Pearson correlation: {corr:.6f}")
else:
    corr = 0
    print("WARNING: too few non-zero pixels to compute correlation")

rmse = np.sqrt(np.mean((our_proj_2d - tigre_proj_0) ** 2))
nrmse = rmse / (tigre_proj_0.max() + 1e-8)
print(f"RMSE: {rmse:.6f}, NRMSE: {nrmse:.4f}")

# ---------- 尝试各种轴变换找到正确映射 ----------
print(f"\n=== 自动搜索正确的轴映射 ===")
import itertools
best_corr = -1
best_transform = None
for perm in itertools.permutations([0, 1, 2]):
    for signs in itertools.product([1, -1], repeat=3):
        trial = our_proj_2d.copy()
        # 对 2D 投影做变换（只有 flipud/fliplr/transpose 有意义）
        # 但这里我们直接在 3D 坐标层面搜索
        pass

# 简化：只搜 2D 投影层面的 8 种变换
transforms_2d = {
    "原始": our_proj_2d,
    "flipud": np.flipud(our_proj_2d),
    "fliplr": np.fliplr(our_proj_2d),
    "flipud+fliplr": np.flipud(np.fliplr(our_proj_2d)),
    "transpose": our_proj_2d.T,
    "transpose+flipud": np.flipud(our_proj_2d.T),
    "transpose+fliplr": np.fliplr(our_proj_2d.T),
    "transpose+flipud+fliplr": np.flipud(np.fliplr(our_proj_2d.T)),
}

print(f"{'变换':<25} {'Corr':>8} {'NRMSE':>8}")
print("-" * 45)
for name, trial in transforms_2d.items():
    if mask.sum() > 10:
        c = np.corrcoef(trial[mask], tigre_proj_0[mask])[0, 1]
    else:
        c = 0
    r = np.sqrt(np.mean((trial - tigre_proj_0) ** 2)) / (tigre_proj_0.max() + 1e-8)
    print(f"{name:<25} {c:>8.4f} {r:>8.4f}")
    if c > best_corr:
        best_corr = c
        best_transform = name

print(f"\n最佳变换: {best_transform} (corr={best_corr:.4f})")

# ---------- 保存热力图对比 ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(tigre_proj_0, cmap="hot")
axes[0, 0].set_title("TIGRE Ax() (GT)")
axes[0, 1].imshow(our_proj_2d, cmap="hot")
axes[0, 1].set_title(f"ray_utils (原始)\ncorr={corr:.4f}")
axes[0, 2].imshow(stored_proj_0, cmap="hot")
axes[0, 2].set_title("存储的 projections[0]")

# 最佳变换
best_proj = transforms_2d[best_transform]
best_nrmse = np.sqrt(np.mean((best_proj - tigre_proj_0)**2)) / (tigre_proj_0.max() + 1e-8)
axes[1, 0].imshow(best_proj, cmap="hot")
axes[1, 0].set_title(f"ray_utils ({best_transform})\ncorr={best_corr:.4f}")
axes[1, 1].imshow(np.abs(best_proj - tigre_proj_0), cmap="hot")
axes[1, 1].set_title(f"差异图 (NRMSE={best_nrmse:.4f})")

# 散点图
axes[1, 2].scatter(tigre_proj_0.flatten()[::100], best_proj.flatten()[::100], s=1, alpha=0.3)
axes[1, 2].plot([0, tigre_proj_0.max()], [0, tigre_proj_0.max()], "r--")
axes[1, 2].set_xlabel("TIGRE")
axes[1, 2].set_ylabel("ray_utils")
axes[1, 2].set_title("散点图 (采样)")

plt.tight_layout()
plt.savefig("outputs/diagnose_coords.png", dpi=150)
print(f"\n热力图已保存到 outputs/diagnose_coords.png")

# ---------- 额外诊断：去掉 _to_tigre_world 看看 ----------
print(f"\n=== 诊断：去掉 _to_tigre_world 的效果 ===")
# 手动重新生成射线，不用 _to_tigre_world
from anatcoder.models.ray_utils import _to_tigre_world
import torch.nn.functional as F

angle_val = float(angles[0])
cos_a = np.cos(angle_val)
sin_a = np.sin(angle_val)
rows, cols = 256, 256
d_row, d_col = 1.5, 1.5

source_raw = torch.tensor([-1000*sin_a, 1000*cos_a, 0.0], dtype=torch.float32)
center_radius = 1000.0 - 1536.0  # = -536
det_center_raw = torch.tensor([-center_radius*sin_a, center_radius*cos_a, 0.0], dtype=torch.float32)
det_u = torch.tensor([cos_a, sin_a, 0.0], dtype=torch.float32)
det_v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

row_ids = torch.arange(rows, dtype=torch.float32)
col_ids = torch.arange(cols, dtype=torch.float32)
row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")
row_offsets = ((rows - 1.0) * 0.5 - row_grid) * d_row
col_offsets = (col_grid - (cols - 1.0) * 0.5) * d_col

det_points = (
    det_center_raw[None, None, :]
    + col_offsets[..., None] * det_u[None, None, :]
    + row_offsets[..., None] * det_v[None, None, :]
)
origins_raw = source_raw[None, None, :].expand(rows, cols, 3).reshape(-1, 3)
dirs_raw = F.normalize((det_points - source_raw[None, None, :]).reshape(-1, 3), dim=-1)

# 不做 _to_tigre_world，直接采样
points_raw, steps_raw = sample_points_along_rays(
    origins_raw, dirs_raw, n_samples=512, near=near, far=far, perturb=False
)
pn_raw = normalize_coords(points_raw, volume_size_mm)
grid_raw = (pn_raw * 2.0 - 1.0)[..., [2, 1, 0]]
flat_raw = grid_raw.reshape(1, 1, n_rays * 512, 1, 3)
s_raw = torch.nn.functional.grid_sample(
    phantom_tensor, flat_raw, mode="bilinear", padding_mode="zeros", align_corners=True
)
d_raw = s_raw.reshape(n_rays, 512, 1)
proj_no_transform = renderer(d_raw, steps_raw).detach().numpy().reshape(256, 256)

# 对比
print(f"{'变换':<25} {'Corr':>8} {'NRMSE':>8}")
print("-" * 45)
for name, trial in {
    "有_to_tigre_world": our_proj_2d,
    "无_to_tigre_world": proj_no_transform,
}.items():
    if mask.sum() > 10:
        c = np.corrcoef(trial[mask], tigre_proj_0[mask])[0, 1]
    else:
        c = 0
    r = np.sqrt(np.mean((trial - tigre_proj_0)**2)) / (tigre_proj_0.max() + 1e-8)
    print(f"{name:<25} {c:>8.4f} {r:>8.4f}")

# 无变换版本也搜 2D 翻转
print(f"\n=== 无_to_tigre_world + 2D翻转搜索 ===")
transforms_raw = {
    "raw_原始": proj_no_transform,
    "raw_flipud": np.flipud(proj_no_transform),
    "raw_fliplr": np.fliplr(proj_no_transform),
    "raw_flipud+fliplr": np.flipud(np.fliplr(proj_no_transform)),
    "raw_transpose": proj_no_transform.T,
    "raw_T+flipud": np.flipud(proj_no_transform.T),
    "raw_T+fliplr": np.fliplr(proj_no_transform.T),
    "raw_T+flipud+fliplr": np.flipud(np.fliplr(proj_no_transform.T)),
}
best2_corr = -1
best2_name = None
print(f"{'变换':<25} {'Corr':>8} {'NRMSE':>8}")
print("-" * 45)
for name, trial in transforms_raw.items():
    if mask.sum() > 10:
        c = np.corrcoef(trial[mask], tigre_proj_0[mask])[0, 1]
    else:
        c = 0
    r = np.sqrt(np.mean((trial - tigre_proj_0)**2)) / (tigre_proj_0.max() + 1e-8)
    print(f"{name:<25} {c:>8.4f} {r:>8.4f}")
    if c > best2_corr:
        best2_corr = c
        best2_name = name

print(f"\n无变换最佳: {best2_name} (corr={best2_corr:.4f})")
print(f"有变换最佳: {best_transform} (corr={best_corr:.4f})")

if best2_corr > best_corr:
    print("\n>>> 结论: _to_tigre_world 是有害的！应该去掉。")
    print(f">>> 去掉后最佳 2D 映射: {best2_name}")
elif best_corr > 0.95:
    print("\n>>> 结论: _to_tigre_world 有效，投影对齐良好。问题在别处。")
else:
    print(f"\n>>> 结论: 两种情况 corr 都不高，射线生成逻辑本身可能有错。")
    print(f">>> 有变换 best corr: {best_corr:.4f}")
    print(f">>> 无变换 best corr: {best2_corr:.4f}")

# ---------- 保存第二张对比图 ----------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
axes2[0].imshow(tigre_proj_0, cmap="hot")
axes2[0].set_title("TIGRE (GT)")
axes2[1].imshow(our_proj_2d, cmap="hot")
axes2[1].set_title(f"有 _to_tigre_world\ncorr={corr:.4f}")
axes2[2].imshow(proj_no_transform, cmap="hot")
axes2[2].set_title(f"无 _to_tigre_world")
plt.tight_layout()
plt.savefig("outputs/diagnose_with_vs_without_transform.png", dpi=150)
print(f"对比图已保存到 outputs/diagnose_with_vs_without_transform.png")
