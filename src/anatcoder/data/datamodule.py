"""Lightning DataModule wrapper for sparse-view reconstruction data."""

from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from .dataset import CTReconDataset


class CTReconDataModule(L.LightningDataModule):
    """Construct train/val/test datasets and dataloaders for CT reconstruction."""

    def __init__(self, data_cfg: Any) -> None:
        """Store data configuration and defer dataset instantiation to setup."""
        super().__init__()
        self.data_cfg = data_cfg
        self.data_dir = Path(data_cfg.data_dir)
        self.proj_dir = Path(data_cfg.proj_dir)
        self.train_dataset: CTReconDataset | None = None
        self.val_dataset: CTReconDataset | None = None
        self.test_dataset: CTReconDataset | None = None

    def prepare_data(self) -> None:
        """Download or verify data artifacts if needed."""
        return None

    def setup(self, stage: str | None = None) -> None:
        """Create dataset objects for the specified stage."""
        raise NotImplementedError("TODO: implement datamodule setup")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        raise NotImplementedError("TODO: implement training dataloader")

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        raise NotImplementedError("TODO: implement validation dataloader")

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        raise NotImplementedError("TODO: implement test dataloader")
