"""This module contains classes related to pytorch lightning dataset."""

from pathlib import Path

import lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class WeatherDataModule(pl.LightningDataModule):
    """PyTorch Lighting data module."""

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        train_transforms: v2.Compose | None,
        val_transforms: v2.Compose | None,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms  # for valid & test data
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.train_ds: None | ImageFolder = None
        self.val_ds: None | ImageFolder = None
        self.test_ds: None | ImageFolder = None

        self.num_cls: None | int = None
        self.cls_names: list[str] = []

    def prepare_data(self) -> None:
        # Don't assign a state here!
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_dpath = Path(self.data_dir) / "train"
            self.train_ds = ImageFolder(train_dpath, transform=self.train_transforms)
            self.cls_names = self.train_ds.classes
            self.num_cls = len(self.cls_names)

            val_dpath = Path(self.data_dir) / "val"
            self.val_ds = ImageFolder(val_dpath, transform=self.val_transforms)

        if stage == "test":
            test_dpath = Path(self.data_dir) / "test"
            self.test_ds = ImageFolder(test_dpath, transform=self.val_transforms)

        if stage == "predict":
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> None:
        pass
