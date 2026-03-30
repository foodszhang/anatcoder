"""AnatCoder training entry point.

Usage:
    # 训练 vanilla baseline (50 views)
    python train.py method=vanilla_inr data.n_views=50

    # 训练 AnatCoder
    python train.py method=anatcoder data.n_views=50

    # Debug 模式（快速验证）
    python train.py method=vanilla_inr data=debug trainer=debug

    # 扫参: 多视角对比
    python train.py -m method=vanilla_inr data.n_views=10,20,30,50
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run training and evaluation from a Hydra configuration."""
    # 1. Seed
    # 2. 根据 cfg.method.name 实例化对应 LightningModule
    # 3. 实例化 DataModule
    # 4. 实例化 Trainer
    # 5. trainer.fit(module, datamodule)
    # 6. trainer.test(module, datamodule)
    raise NotImplementedError("TODO: implement training pipeline")


if __name__ == "__main__":
    main()
