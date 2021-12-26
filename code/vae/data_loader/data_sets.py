import torch
from torchvision import datasets, transforms

from typing import List, Tuple, Union

from base import BaseDataset

class UnsupervisedDataset(BaseDataset):
    """A decorator for datasets used in unsupervised settings.
    Allows to compare the input of a network with its reconstructed output.

    Args:
        dataset (BaseDataset): [description]
    """
    def __init__(self, dataset: BaseDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx: int):
        # we re
        x = self.dataset[idx][0]
        return x, x

    def __len__(self):
        return len(self.dataset)