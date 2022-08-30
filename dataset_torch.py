from typing import Tuple

import torch
from torch.utils.data import Dataset

from data_per_day import DataPerDay


class DataPerDayTochDataset(Dataset):
    def __init__(
        self,
        dataperday_dataset: DataPerDay,
        transforms_X=None,
        transforms_agg=None,
        transforms_all=None,
    ):

        super().__init__()
        self.data_elec = torch.as_tensor(
            dataperday_dataset.data_elec, dtype=torch.float32
        ).unsqueeze(1)
        self.data_agg = torch.as_tensor(
            dataperday_dataset.data_agg, dtype=torch.float32
        ).unsqueeze(1)
        self.metadata = dataperday_dataset.y
        self.transforms_all = transforms_all
        self.transforms_X = transforms_X
        self.transforms_agg = transforms_agg

        self.labels = dataperday_dataset.labels()

    def __len__(self) -> int:
        return self.data_elec.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        X = self.data_elec[idx]
        agg = self.data_agg[idx]
        labels = self.labels[idx]
        if self.transforms_X is not None:
            X = self.transforms_X(X)
        if self.transforms_agg is not None:
            agg = self.transforms_agg(agg)

        if self.transforms_all is not None:
            # junto el X e y en la dimension de los channels para aplicar
            # una transformacion 1d acorde
            sample = torch.cat((X, agg), dim=0)
            sample = self.transforms_all(sample)
            X = sample[0 : X.shape[0]]
            agg = sample[X.shape[0] : sample.shape[0]]

        return {"X": X, "agg": agg, "labels": labels, "idx": idx}

    def get_labels(self):
        return self.labels
