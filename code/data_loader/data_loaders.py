import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import torchvision as vis
import torchaudio as audio

from base import BaseDataLoader
from .datasets import VSCO2, YESNOPacked

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = vis.transforms.Compose([
            vis.transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = vis.datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BinaryMnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        trsfm = vis.transforms.Compose([
            vis.transforms.ToTensor(),
            vis.transforms.Lambda(lambda x: x.round())  # to black and white
        ])
        self.dataset = vis.datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class YesNoSpeechDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = YESNOPacked(
            self.data_dir, train=training, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Vsco2DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = VSCO2(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)