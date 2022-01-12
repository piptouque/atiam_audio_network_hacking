import numpy as np
import torch
from torch.utils.data import ConcatDataset
import torchvision as vis
from typing import List, Tuple, Union, Any

from base import BaseDataLoader, GenerativeModel
from .datasets import VSCO2, YESNOPacked, GeneratedDataset


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
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=audio_collate_fn)


class Vsco2DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = VSCO2(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=audio_collate_fn)


class AdversarialDataloader(BaseDataLoader):
    def __init__(self, gen_data_loader: BaseDataLoader, gen_model: GenerativeModel) -> None:
        self._gen_model = gen_model
        self._gen_data_loader = gen_data_loader
        self._gen_dataset = self._gen_data_loader.dataset
        self._dis_dataset = GeneratedDataset(
            self._gen_model, len(self._gen_dataset))
        self.dataset = ConcatDataset([self._gen_dataset, self._dis_dataset])
        super().__init__(self.dataset, self._gen_data_loader.batch_size, self._gen_data_loader.shuffle,
                         self._gen_data_loader.validation_split, self._gen_data_loader.num_workers)
