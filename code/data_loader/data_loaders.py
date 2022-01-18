import numpy as np
import torch
from torch.utils.data import ConcatDataset
import torchvision as vis
from typing import List, Tuple, Union, Any

from base import BaseDataLoader, GenerativeModel
from .datasets import VSCO2, YESNOPacked, GeneratedDataset, UntamperedDataset


class MnistDataLoader(BaseDataLoader):
    """MNIST data loader, with pixel values in [0, 1]
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
    """MNIST data loader, with pixel values equal to 0 or 1.
    """

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
    """Packs audio data and sample rate in a tuple.

    Args:
        data (List[Tuple[Any]]): Input audio data, sample rate and label

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Output data and label
    """
    data, label = zip(*data)
    audio, sr = zip(*data)
    audio = torch.tensor(audio)
    sr = torch.tensor(sr)
    label = torch.tensor(label)
    return audio, sr, label


class YesNoSpeechDataLoader(BaseDataLoader):
    """Data loader for the YESNO dataset.

    """

    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = YESNOPacked(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=audio_collate_fn)


class Vsco2DataLoader(BaseDataLoader):
    """Data loader for the VSCO2 dataset.

    """

    def __init__(self, data_dir: str, batch_size: int, transform=None, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        self.dataset = VSCO2(
            self.data_dir, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=audio_collate_fn)


class AdversarialDataloader(BaseDataLoader):
    """Data loader for adversarial training.
    Creates the discriminator dataset from the generator and the generator's data loader.

    """

    def __init__(self, gen_data_loader: BaseDataLoader, gen_model: GenerativeModel) -> None:
        self._gen_model = gen_model
        self._gen_data_loader = gen_data_loader
        self._gen_dataset = UntamperedDataset(self._gen_data_loader.dataset)

        self._dis_dataset = GeneratedDataset(
            self._gen_model, len(self._gen_dataset))
        self.dataset = ConcatDataset([self._gen_dataset, self._dis_dataset])
        super().__init__(self.dataset, self._gen_data_loader.batch_size, self._gen_data_loader.shuffle,
                         self._gen_data_loader.validation_split, self._gen_data_loader.num_workers)
