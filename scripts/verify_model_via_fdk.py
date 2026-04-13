import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tigre
from tigre.algorithms import fdk

from anatcoder.models.network import VanillaINR
from anatcoder.models.ray_utils import generate_rays_for_view, normalize_coords
from anatcoder.models.renderer import reconstruct_volume
from anatcoder.utils.geometry import CBCTGeometry


# ============================================================
# 0. 加载模型
# ============================================================
ckpt_candidates = sorted(Path("logs").rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
if not ckpt_candidates:
    raise FileNotFoundError("No checkpoint found under logs/**.ckpt")
ckpt_path = ckpt_candidates[-1]
print(f"Checkpoint: {ckpt_path}")

state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
ms = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}

# 自动推断 encoder 维度
enc_dim = None
for k in sorted(ms.keys()):
    if k.endswith("mlp.0.weight"):
        enc_dim = int(ms[k].shape[1])
        break
if enc_dim is None:
    enc_dim = 32
print(f"Inferred encoder dim: {enc_dim}")

encoder_type = "positional" if enc_dim == 63 else "hashgrid"
model = VanillaINR(encoder_type=encoder_type, n_hidden_layers=4, hidden_dim=256)
load_result = model.load_state_dict(ms, strict=False)
print(
    f"Encoder type: {encoder_type}, "
    f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
)
model.volume_size_mm = [128.0] * 3
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Device: {device}")


# ============================================================
# 1. 加载 GT 和几何
# ============================================================
gt_vol = np.load("data/processed/case001/volume.npy").astype(np.float32)
geo = CBCTGeometry()
tigre_geo = geo.to_tigre_geometry()

N = int(gt_vol.shape[0])
vol_mm = [128.0, 128.0, 128.0]
diag = float(np.linalg.norm(vol_mm))
near = float(geo.DSO - 0.5 * diag)
far = float(geo.DSO + 0.5 * diag)

print(f"GT volume shape: {gt_vol.shape}")
print(f"near={near:.2f}, far={far:.2f}")


# ============================================================
# 2. 用模型渲染 360 度投影
# ============================================================
n_render_views = 360
render_angles = np.linspace(0, 2 * np.pi, n_render_views, endpoint=False).astype(np.float32)
det_rows, det_cols = int(geo.n_detector[0]), int(geo.n_detector[1])
n_samples = 128
batch_size = 8192

print(f"\n渲染 {n_render_views} 个角度的投影...")
rendered_projections = np.zeros((n_render_views, det_rows, det_cols), dtype=np.float32)

with torch.no_grad():
    for vi in range(n_render_views):
        angle = float(render_angles[vi])
        origins, directions = generate_rays_for_view(geo, angle=angle, device=device)

        n_rays = origins.shape[0]
        proj_flat = torch.zeros(n_rays, device=device, dtype=torch.float32)
        step_size = (far - near) / n_samples

        for start in range(0, n_rays, batch_size):
            end = min(start + batch_size, n_rays)
            o_batch = origins[start:end]
            d_batch = directions[start:end]

            t_vals = torch.linspace(near, far, n_samples, device=device, dtype=torch.float32)
            t_vals = t_vals.unsqueeze(0).expand(end - start, -1)
            pts = o_batch.unsqueeze(1) + d_batch.unsqueeze(1) * t_vals.unsqueeze(-1)
            pts_norm = normalize_coords(pts.reshape(-1, 3), vol_mm)

            autocast_ctx = (
                torch.cuda.amp.autocast if device.type == "cuda" else nullcontext  # type: ignore[assignment]
            )
            with autocast_ctx():
                mu = model.query_density(pts_norm).squeeze(-1)

            mu = mu.reshape(end - start, n_samples)
            mu = torch.clamp(mu, min=0.0)
            line_integral = mu.sum(dim=1) * step_size
            proj_flat[start:end] = line_integral.to(torch.float32)

        rendered_projections[vi] = proj_flat.cpu().numpy().reshape(det_rows, det_cols)

        if (vi + 1) % 60 == 0 or vi == 0:
            p = rendered_projections[vi]
            print(
                f"  角度 {vi + 1}/{n_render_views} 完成, "
                f"投影范围: [{p.min():.4f}, {p.max():.4f}]"
            )


# ============================================================
# 3. 对比: 模型渲染投影 vs TIGRE 真实投影（50 views）
# ============================================================
print("\n=== 投影域对比（50 views） ===")
gt_angles = np.load("data/projections/case001/50views/angles.npy").astype(np.float32)
gt_projs = np.load("data/projections/case001/50views/projections.npy").astype(np.float32)

# 从 360 度渲染中找最接近 50 view 角度的
proj_domain_psnrs = []
for i in range(len(gt_angles)):
    angle_diff = np.abs(render_angles - gt_angles[i])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    closest_idx = int(np.argmin(angle_diff))
    rendered = rendered_projections[closest_idx]
    gt_proj = gt_projs[i]
    mse = float(np.mean((rendered - gt_proj) ** 2))
    if mse > 1e-10:
        psnr = 10 * np.log10((float(gt_proj.max()) ** 2 + 1e-10) / mse)
    else:
        psnr = 99.0
    proj_domain_psnrs.append(psnr)

mean_proj_psnr = float(np.mean(proj_domain_psnrs))
print(f"投影域平均 PSNR: {mean_proj_psnr:.2f} dB")
print(f"投影域 PSNR 范围: [{min(proj_domain_psnrs):.2f}, {max(proj_domain_psnrs):.2f}]")

# 还要看相关性
rendered_0 = rendered_projections[0]
gt_0 = gt_projs[0]
corr = float(np.corrcoef(rendered_0.flatten(), gt_0.flatten())[0, 1])
print(f"角度0 投影 Pearson correlation: {corr:.4f}")
print(f"角度0 渲染范围: [{rendered_0.min():.4f}, {rendered_0.max():.4f}]")
print(f"角度0 GT范围:   [{gt_0.min():.4f}, {gt_0.max():.4f}]")
print(f"Scale ratio: {rendered_0.mean() / (gt_0.mean() + 1e-8):.4f}")


# ============================================================
# 4. TIGRE FDK 重建
# ============================================================
print("\n=== TIGRE FDK 重建（从模型渲染的 360 度投影） ===")
fdk_recon = fdk(rendered_projections.astype(np.float32), tigre_geo, render_angles.astype(np.float32))
fdk_recon = np.clip(fdk_recon, 0, None).astype(np.float32)

fdk_mse = float(np.mean((fdk_recon - gt_vol) ** 2))
fdk_psnr = 10 * np.log10(1.0 / (fdk_mse + 1e-10))
print(f"FDK重建 PSNR: {fdk_psnr:.2f} dB")
print(f"FDK重建范围: [{fdk_recon.min():.4f}, {fdk_recon.max():.4f}]")
print(f"GT范围:      [{gt_vol.min():.4f}, {gt_vol.max():.4f}]")

# 对比: 直接用 GT volume 做 360 投影再 FDK（理论上限）
print("\n=== 对照: GT volume → 360投影 → FDK（理论上限） ===")
gt_360_projs = tigre.Ax(gt_vol.astype(np.float32), tigre_geo, render_angles.astype(np.float32))
fdk_from_gt = fdk(gt_360_projs, tigre_geo, render_angles.astype(np.float32))
fdk_from_gt = np.clip(fdk_from_gt, 0, None).astype(np.float32)
gt_fdk_mse = float(np.mean((fdk_from_gt - gt_vol) ** 2))
gt_fdk_psnr = 10 * np.log10(1.0 / (gt_fdk_mse + 1e-10))
print(f"GT→360投影→FDK PSNR: {gt_fdk_psnr:.2f} dB (理论上限)")


# ============================================================
# 5. 也试一下直接用模型 reconstruct_volume 的结果做对比
# ============================================================
print("\n=== 对比: 当前 reconstruct_volume ===")
direct_recon = reconstruct_volume(model, [N, N, N], [1.0, 1.0, 1.0])
direct_mse = float(np.mean((direct_recon - gt_vol) ** 2))
direct_psnr = 10 * np.log10(1.0 / (direct_mse + 1e-10))
print(f"直接 reconstruct_volume PSNR: {direct_psnr:.2f} dB")


# ============================================================
# 6. 汇总
# ============================================================
print(f"\n{'=' * 60}")
print("汇总:")
print(f"  投影域平均 PSNR:              {mean_proj_psnr:.2f} dB")
print(f"  投影域 corr (角度0):          {corr:.4f}")
print(f"  模型渲染→FDK重建 PSNR:       {fdk_psnr:.2f} dB")
print(f"  GT→360投影→FDK PSNR (上限):  {gt_fdk_psnr:.2f} dB")
print(f"  直接 reconstruct_volume PSNR: {direct_psnr:.2f} dB")
print("  FDK 50views baseline:         37.93 dB")
print(f"{'=' * 60}")

if fdk_psnr > 35:
    print("\n>>> 模型渲染→FDK PSNR 高！模型学对了！")
    print(">>> 问题确认在 reconstruct_volume 坐标映射。")
    print(">>> 下一步: 从 FDK 重建反推正确的坐标映射。")
elif fdk_psnr > 30:
    print("\n>>> 模型渲染→FDK PSNR 中等，模型基本学对但有噪声。")
    print(">>> 可能需要更多训练或更好的超参。")
else:
    print("\n>>> 模型渲染→FDK PSNR 也低！模型本身没学好。")
    print(">>> 问题在训练端（renderer/loss/ray generation），不是 reconstruct_volume。")


# ============================================================
# 7. 保存可视化
# ============================================================
Path("outputs").mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
s = N // 2
vmin, vmax = float(gt_vol.min()), float(gt_vol.max())

axes[0, 0].imshow(gt_vol[s], cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("GT axial")
axes[0, 1].imshow(gt_vol[:, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("GT coronal")
axes[0, 2].imshow(gt_vol[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 2].set_title("GT sagittal")

axes[1, 0].imshow(fdk_recon[s], cmap="gray", vmin=vmin, vmax=vmax)
axes[1, 0].set_title(f"Model->FDK axial\nPSNR={fdk_psnr:.2f}")
axes[1, 1].imshow(fdk_recon[:, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Model->FDK coronal")
axes[1, 2].imshow(fdk_recon[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[1, 2].set_title("Model->FDK sagittal")

axes[2, 0].imshow(direct_recon[s], cmap="gray", vmin=vmin, vmax=vmax)
axes[2, 0].set_title(f"Direct recon axial\nPSNR={direct_psnr:.2f}")
axes[2, 1].imshow(direct_recon[:, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[2, 1].set_title("Direct recon coronal")
axes[2, 2].imshow(direct_recon[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
axes[2, 2].set_title("Direct recon sagittal")

plt.suptitle("GT vs Model->FDK vs Direct Reconstruct", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/verify_model_via_fdk.png", dpi=150)
print("\n切片对比已保存到 outputs/verify_model_via_fdk.png")

# 投影对比图
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
axes2[0].imshow(gt_0, cmap="hot")
axes2[0].set_title("GT projection (angle 0)")
axes2[1].imshow(rendered_0, cmap="hot")
axes2[1].set_title(f"Model rendered (angle 0)\ncorr={corr:.4f}")
axes2[2].imshow(np.abs(rendered_0 - gt_0), cmap="hot")
axes2[2].set_title("Difference")
plt.tight_layout()
plt.savefig("outputs/verify_projection_compare.png", dpi=150)
print("投影对比已保存到 outputs/verify_projection_compare.png")

print("\n把终端完整输出和两张 PNG 贴给我。")
