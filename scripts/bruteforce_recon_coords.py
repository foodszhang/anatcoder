"""暴力搜索 reconstruct_volume 的正确坐标映射。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anatcoder.models.ray_utils import normalize_coords, _to_tigre_world
from anatcoder.models.network import VanillaINR

# ---------- 加载模型 ----------
ckpt = sorted(Path("logs").rglob("*.ckpt"))[-1]
print(f"Checkpoint: {ckpt}")
state = torch.load(ckpt, map_location="cpu")["state_dict"]
ms = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}
# 与 checkpoint 结构自动对齐（当前 ckpt 为 hashgrid: in_dim=32）
in_dim = ms["mlp.0.weight"].shape[1]
encoder_type = "positional" if in_dim == 63 else "hashgrid"
model = VanillaINR(encoder_type=encoder_type, n_hidden_layers=4, hidden_dim=256)
model.load_state_dict(ms, strict=False)
model.volume_size_mm = [128.0] * 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Inferred encoder_type: {encoder_type}, device: {device}")

gt = np.load("data/processed/case001/volume.npy").astype(np.float32)
print(f"GT shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")

N = 128
d = 1.0
vol_mm = [128.0, 128.0, 128.0]
batch_size = 65536

# ---------- 构建物理坐标 ----------
coords_1d = torch.linspace(-(N - 1) / 2 * d, (N - 1) / 2 * d, N)
zz, yy, xx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
# zz[i,j,k] = coords_1d[i], yy[i,j,k] = coords_1d[j], xx[i,j,k] = coords_1d[k]

# 基础坐标分量
components = {"x": xx.reshape(-1), "y": yy.reshape(-1), "z": zz.reshape(-1)}

def query_model(coords_flat):
    """coords_flat: [N^3, 3] 归一化坐标 -> 密度"""
    results = []
    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        chunk = coords_flat[start:end].to(device)
        with torch.no_grad():
            if hasattr(model, "query_density"):
                mu = model.query_density(chunk)
            else:
                mu = model(chunk)
        results.append(mu.cpu())
    return torch.cat(results, dim=0)

# ---------- 暴力搜索 ----------
axis_names = ["x", "y", "z"]
axis_perms = list(itertools.permutations(axis_names))
sign_combos = list(itertools.product([1, -1], repeat=3))
reshape_orders = [
    ("zyx", lambda v: v.reshape(N, N, N)),       # 原始 meshgrid 顺序
    ("xyz", lambda v: v.reshape(N, N, N).permute(2, 1, 0).numpy() if isinstance(v, torch.Tensor) else v.reshape(N, N, N).transpose(2, 1, 0)),
]

# 有无 _to_tigre_world
transform_options = [
    ("with_ttw", True),
    ("no_ttw", False),
]

results = []
total = len(axis_perms) * len(sign_combos) * len(transform_options)
print(f"\n总搜索空间: {len(axis_perms)} 排列 x {len(sign_combos)} 符号 x {len(transform_options)} 变换 = {total} 组合")
print(f"{'#':>4} {'轴序':>8} {'符号':>10} {'变换':>10} {'PSNR':>8}")
print("-" * 50)

best_psnr = -999
best_config = None
best_vol = None

count = 0
for perm in axis_perms:
    for signs in sign_combos:
        for tf_name, use_ttw in transform_options:
            count += 1
            
            # 构建坐标: stack 顺序 = perm, 符号 = signs
            c0 = components[perm[0]] * signs[0]
            c1 = components[perm[1]] * signs[1]
            c2 = components[perm[2]] * signs[2]
            phys = torch.stack([c0, c1, c2], dim=-1)
            
            if use_ttw:
                transformed = _to_tigre_world(phys)
            else:
                transformed = phys
            
            normed = normalize_coords(transformed, vol_mm)
            mu = query_model(normed).squeeze(-1)
            
            # reshape 回 [N,N,N]，按 meshgrid 的 (z,y,x) 顺序
            vol = mu.numpy().reshape(N, N, N)
            
            mse = np.mean((vol - gt) ** 2)
            if mse < 1e-10:
                psnr = 99.0
            else:
                psnr = 10 * np.log10(1.0 / mse)
            
            sign_str = f"({signs[0]:+d},{signs[1]:+d},{signs[2]:+d})"
            perm_str = ",".join(perm)
            
            if psnr > best_psnr:
                best_psnr = psnr
                best_config = (perm, signs, tf_name)
                best_vol = vol.copy()
            
            if psnr > 27:
                print(f"{count:>4} {perm_str:>8} {sign_str:>10} {tf_name:>10} {psnr:>8.2f} ***")
                results.append((psnr, perm_str, sign_str, tf_name))
            elif count % 24 == 0:
                print(f"{count:>4} {perm_str:>8} {sign_str:>10} {tf_name:>10} {psnr:>8.2f}")

# ---------- 也搜 reshape 后 transpose ----------
print(f"\n=== 额外: 对最优配置尝试 volume transpose ===")
if best_vol is not None:
    transpose_options = {
        "原始 [z,y,x]": best_vol,
        "transpose(2,1,0) [x,y,z]": best_vol.transpose(2, 1, 0),
        "transpose(0,2,1) [z,x,y]": best_vol.transpose(0, 2, 1),
        "transpose(1,0,2) [y,z,x]": best_vol.transpose(1, 0, 2),
        "transpose(1,2,0) [y,x,z]": best_vol.transpose(1, 2, 0),
        "transpose(2,0,1) [x,z,y]": best_vol.transpose(2, 0, 1),
    }
    for name, tvol in transpose_options.items():
        mse_t = np.mean((tvol - gt) ** 2)
        psnr_t = 10 * np.log10(1.0 / (mse_t + 1e-10))
        marker = " ***" if psnr_t > best_psnr else ""
        print(f"  {name:<30} PSNR={psnr_t:.2f}{marker}")
        if psnr_t > best_psnr:
            best_psnr = psnr_t
            best_vol = tvol.copy()
            best_config = (*best_config, name)

# ---------- 汇总 ----------
print(f"\n{'='*50}")
print(f"最优配置: {best_config}")
print(f"最优 PSNR: {best_psnr:.2f} dB")
print(f"{'='*50}")

# ---------- Top 10 ----------
results.sort(reverse=True)
print(f"\nPSNR > 27 的所有配置:")
for psnr, perm, sign, tf in results[:20]:
    print(f"  PSNR={psnr:.2f}  axes=({perm}) signs={sign} transform={tf}")

# ---------- 保存可视化 ----------
if best_vol is not None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    s = N // 2
    vmin, vmax = gt.min(), gt.max()
    axes[0, 0].imshow(gt[s, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("GT axial")
    axes[0, 1].imshow(gt[:, s, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("GT coronal")
    axes[0, 2].imshow(gt[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("GT sagittal")
    axes[1, 0].imshow(best_vol[s, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"Best recon axial\nPSNR={best_psnr:.2f}")
    axes[1, 1].imshow(best_vol[:, s, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Best recon coronal")
    axes[1, 2].imshow(best_vol[:, :, s], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("Best recon sagittal")
    plt.suptitle(f"Best config: {best_config}", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/bruteforce_recon_best.png", dpi=150)
    print(f"\n最优重建切片已保存到 outputs/bruteforce_recon_best.png")

print("\n完成。把终端完整输出和 bruteforce_recon_best.png 贴给我。")
