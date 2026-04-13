import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 加载模型 ----------
from anatcoder.models.network import VanillaINR

ckpt_path = sorted(Path("logs").rglob("*.ckpt"))[-1]
print(f"Checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
ms = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}

sd_keys = list(ms.keys())
enc_in_dim = None
for k in sd_keys:
    if "backbone" in k and "0" in k and "weight" in k:
        enc_in_dim = ms[k].shape[1]
        break
if enc_in_dim is None:
    # 兼容当前 VanillaINR: first layer lives in mlp.0.weight
    if "mlp.0.weight" in ms:
        enc_in_dim = ms["mlp.0.weight"].shape[1]
    else:
        enc_in_dim = 32
print(f"Inferred encoder output dim: {enc_in_dim}")

# 自动推断 encoder 类型
encoder_type = "positional" if enc_in_dim == 63 else "hashgrid"
model = VanillaINR(encoder_type=encoder_type, n_hidden_layers=4, hidden_dim=256)
load_result = model.load_state_dict(ms, strict=False)
print(f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
model.volume_size_mm = [128.0] * 3
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------- 加载 GT ----------
gt_np = np.load("data/processed/case001/volume.npy").astype(np.float32)
gt = torch.from_numpy(gt_np).to(device)
print(f"GT shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")

N = 128
d = 1.0

# ---------- 构建物理坐标 ----------
coords_1d = torch.linspace(-(N - 1) / 2 * d, (N - 1) / 2 * d, N, device=device)
zz, yy, xx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
phys_flat = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
print(f"Physical coords shape: {phys_flat.shape}")

vol_mm = torch.tensor([128.0, 128.0, 128.0], device=device)

def normalize_coords_torch(coords, vol_mm):
    return coords / vol_mm + 0.5

def query_model_batched(model, normalized, batch_size=65536):
    mu_all = []
    for start in range(0, normalized.shape[0], batch_size):
        end = min(start + batch_size, normalized.shape[0])
        chunk = normalized[start:end]
        with torch.no_grad():
            if hasattr(model, "query_density"):
                mu = model.query_density(chunk)
            else:
                mu = model(chunk)
        mu_all.append(mu)
    return torch.cat(mu_all, dim=0).squeeze(-1)

# ---------- 先验证当前最优离散映射 ----------
print("\n=== 验证离散最优 (-y, -x, -z) ===")
discrete_best = torch.stack([-yy, -xx, -zz], dim=-1).reshape(-1, 3)
discrete_norm = normalize_coords_torch(discrete_best, vol_mm)
discrete_vol = query_model_batched(model, discrete_norm).reshape(N, N, N)
discrete_mse = F.mse_loss(discrete_vol, gt).item()
discrete_psnr = 10 * np.log10(1.0 / (discrete_mse + 1e-10))
print(f"Discrete best PSNR: {discrete_psnr:.2f} dB")

# ---------- 仿射变换优化 ----------
print("\n=== 开始仿射优化 ===")

A = torch.eye(3, device=device, requires_grad=True)
b = torch.zeros(3, device=device, requires_grad=True)

optimizer = torch.optim.Adam([A, b], lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)

n_sample_points = 50000
best_psnr = -999
best_A = A.data.clone()
best_b = b.data.clone()

for step in range(500):
    optimizer.zero_grad()

    idx = torch.randint(0, phys_flat.shape[0], (n_sample_points,), device=device)
    phys_sample = phys_flat[idx]
    gt_sample = gt.reshape(-1)[idx]

    transformed = phys_sample @ A.T + b
    normalized = normalize_coords_torch(transformed, vol_mm)

    if hasattr(model, "query_density"):
        mu = model.query_density(normalized).squeeze(-1)
    else:
        mu = model(normalized).squeeze(-1)

    loss = F.mse_loss(mu, gt_sample)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (step + 1) % 50 == 0 or step == 0:
        psnr = 10 * np.log10(1.0 / (loss.item() + 1e-10))
        det = torch.det(A).item()
        print(f"Step {step+1:>4}: loss={loss.item():.6f} PSNR~{psnr:.2f} det(A)={det:.4f} lr={scheduler.get_last_lr()[0]:.5f}")

        if psnr > best_psnr:
            best_psnr = psnr
            best_A = A.data.clone()
            best_b = b.data.clone()

# ---------- 用最优仿射做全量重建 ----------
print(f"\n=== 最优仿射变换 ===")
print(f"A =\n{best_A.cpu().numpy()}")
print(f"b = {best_b.cpu().numpy()}")
print(f"det(A) = {torch.det(best_A).item():.6f}")

full_transformed = phys_flat @ best_A.T + best_b
full_normalized = normalize_coords_torch(full_transformed, vol_mm)
full_vol = query_model_batched(model, full_normalized).reshape(N, N, N)

full_mse = F.mse_loss(full_vol, gt).item()
full_psnr = 10 * np.log10(1.0 / (full_mse + 1e-10))
print(f"\n离散最优 PSNR: {discrete_psnr:.2f} dB")
print(f"仿射优化 PSNR: {full_psnr:.2f} dB")
print(f"提升: {full_psnr - discrete_psnr:.2f} dB")

if full_psnr > 32:
    print("\n>>> 仿射优化大幅提升！坐标映射确实需要连续变换。")
    print(">>> 接下来需要把 A 和 b 反推回物理含义，固化到代码。")
elif full_psnr > 29:
    print("\n>>> 有改善但不够大，可能还有 scale/offset 问题。")
else:
    print("\n>>> 仿射优化也没用，问题不在坐标映射。")
    print(">>> 可能是 renderer 或 encoder 问题。")

# ---------- 分析 A 矩阵含义 ----------
print(f"\n=== A 矩阵分析 ===")
U, S, Vt = torch.linalg.svd(best_A)
print(f"奇异值: {S.cpu().numpy()}")
print(f"U (旋转1):\n{U.cpu().numpy()}")
print(f"Vt (旋转2):\n{Vt.cpu().numpy()}")

is_perm = True
A_np = best_A.cpu().numpy()
for i in range(3):
    row_max = np.max(np.abs(A_np[i]))
    row_nnz = np.sum(np.abs(A_np[i]) > 0.1 * row_max)
    if row_nnz != 1:
        is_perm = False

if is_perm:
    print("\nA 近似一个轴排列矩阵（带符号）:")
    for i in range(3):
        j = np.argmax(np.abs(A_np[i]))
        sign = "+" if A_np[i, j] > 0 else "-"
        axis = ["x", "y", "z"][j]
        print(f"  输出 dim{i} = {sign}{axis} (系数={A_np[i,j]:.4f})")
else:
    print("\nA 不是简单轴排列，包含旋转/缩放分量")
    print("这说明 ray_utils 的坐标系和 TIGRE 之间有非平凡的旋转关系")

# ---------- 保存可视化 ----------
fig, axes_plt = plt.subplots(2, 3, figsize=(12, 8))
s = N // 2
vmin, vmax = gt_np.min(), gt_np.max()
axes_plt[0, 0].imshow(gt_np[s, :, :], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[0, 0].set_title("GT axial")
axes_plt[0, 1].imshow(gt_np[:, s, :], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[0, 1].set_title("GT coronal")
axes_plt[0, 2].imshow(gt_np[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[0, 2].set_title("GT sagittal")

vol_np = full_vol.cpu().numpy()
axes_plt[1, 0].imshow(vol_np[s, :, :], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[1, 0].set_title(f"Affine recon axial\nPSNR={full_psnr:.2f}")
axes_plt[1, 1].imshow(vol_np[:, s, :], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[1, 1].set_title("Affine recon coronal")
axes_plt[1, 2].imshow(vol_np[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
axes_plt[1, 2].set_title("Affine recon sagittal")

plt.suptitle(f"Discrete: {discrete_psnr:.2f} dB -> Affine: {full_psnr:.2f} dB", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/affine_recon.png", dpi=150)
print(f"\n切片对比已保存到 outputs/affine_recon.png")
print(f"\n把终端完整输出和 affine_recon.png 贴给我。")
