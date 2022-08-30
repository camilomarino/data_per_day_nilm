from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from transforms_1d_torch import transforms1d

from data_per_day import DataPerDay
from dataset_torch import DataPerDayTochDataset


class DataPerDayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        elec_name: Optional[str] = None,
        normalize: bool = True,
        transforms_train_X=nn.Identity(),
        transforms_train_agg=nn.Identity(),
        transforms_train_all=nn.Identity(),
        balance_sampler=True,
        batch_size: int = 64,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.elec_name = elec_name
        self.normalize = normalize
        self.transforms_train_X = transforms_train_X
        self.transforms_train_agg = transforms_train_agg
        self.transforms_train_all = transforms_train_all
        self.balance_sampler = balance_sampler
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_data_per_day = self._load_data_per_day(self.train_path)
        self.val_data_per_day = self._load_data_per_day(self.val_path)
        self.test_data_per_day = self._load_data_per_day(self.test_path)

        if self.normalize:
            self._add_normalize_transform()
        else:
            self.transforms_val_X = nn.Identity()
            self.transforms_val_agg = nn.Identity()
            self.transforms_test_X = nn.Identity()
            self.transforms_test_agg = nn.Identity()

        self.train_dataset = DataPerDayTochDataset(
            self.train_data_per_day,
            transforms_X=self.transforms_train_X,
            transforms_agg=self.transforms_train_agg,
            transforms_all=self.transforms_train_all,
        )
        self.val_dataset = DataPerDayTochDataset(
            self.val_data_per_day,
            transforms_X=self.transforms_val_X,
            transforms_agg=self.transforms_val_agg,
        )
        self.test_dataset = DataPerDayTochDataset(
            self.test_data_per_day,
            transforms_X=self.transforms_test_X,
            transforms_agg=self.transforms_test_agg,
        )

    def _load_data_per_day(self, path: Path):
        """
        Load dataperday from path and clean it.
        """
        dataperday = (
            DataPerDay.load(path)
            .drop_aggregate_nan()
            .clean_signals()
            .get_elec(self.elec_name)
        )
        return dataperday

    def _add_normalize_transform(self):
        self.mu_X, self.std_X = (
            self.train_data_per_day.X.mean(),
            self.train_data_per_day.X.std(),
        )
        self.mu_agg, self.std_agg = (
            self.train_data_per_day.agg.mean(),
            self.train_data_per_day.agg.std(),
        )
        scale_X_transform = transforms1d.Scale(self.mu_X, self.std_X)
        scale_agg_transform = transforms1d.Scale(self.mu_agg, self.std_agg)

        self.transforms_train_X = transforms1d.Compose(
            [scale_X_transform, self.transforms_train_X]
        )
        self.transforms_train_agg = transforms1d.Compose(
            [scale_agg_transform, self.transforms_train_agg]
        )
        self.transforms_val_X = scale_X_transform
        self.transforms_val_agg = scale_agg_transform
        self.transforms_test_X = scale_X_transform
        self.transforms_test_agg = scale_agg_transform

    def get_scale_values(self):
        return {
            "mu_X": self.mu_X,
            "std_X": self.std_X,
            "mu_agg": self.mu_agg,
            "std_agg": self.std_agg,
        }

    def train_dataloader(self):
        if self.balance_sampler:
            sampler = ImbalancedDatasetSampler(self.train_dataset)
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=sampler
            )
        else:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
