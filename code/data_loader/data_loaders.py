import numpy as np
from typing import List, Tuple, Union, Any
import torch

import torchvision as vis

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
            vis.transforms.Lambda(np.around)  # to black and white
        ])
        self.dataset = vis.datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def audio_collate_fn(data: List[Tuple[Any]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        audio, sr, label = zip(*data)
        audio = torch.tensor(audio)
        sr = torch.tensor(sr)
        label = torch.tensor(label)
        return audio, sr, label

class YesNoSpeechDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = YESNOPacked(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=audio_collate_fn)


class Vsco2DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = VSCO2(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=audio_collate_fn)
