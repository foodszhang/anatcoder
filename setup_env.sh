#!/bin/bash
set -e

echo "=== AnatCoder Environment Setup ==="
echo "Using uv for dependency management"

# 1. 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. 创建虚拟环境 + 安装基础依赖
echo "Creating virtual environment and installing dependencies..."
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. 安装 CUDA 相关依赖（需要 CUDA toolkit 已安装）
echo "Installing TIGRE v3..."
uv pip install tigre

echo "Installing tiny-cuda-nn (需要 CUDA toolkit + cmake)..."
uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 4. 验证安装
echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning {lightning.__version__}')"
python -c "import hydra; print(f'Hydra {hydra.__version__}')"
python -c "import tigre; print(f'TIGRE installed')"
python -c "import tinycudann; print(f'tiny-cuda-nn installed')" 2>/dev/null || echo "WARN: tinycudann not available (needs GPU)"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source .venv/bin/activate"
echo "Quick test:    python -m pytest tests/ -v"
