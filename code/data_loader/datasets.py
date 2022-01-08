
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio 

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchaudio.datasets.yesno import _RELEASE_CONFIGS as _YESNO_RELEASE_CONFIGS
from torchaudio.datasets import YESNO

from utils.download_util import extract_archive

_VSCO2_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "waves_yesno",
        "url": "https://github.com/sgossner/VSCO-2-CE/archive/refs/heads/master.zip",
        "checksum": "c3f49e0cca421f96b75b41640749167b52118f232498667ca7a5f9416aef8e73",
    }
}

class VSCO2(Dataset):
    """Create a Dataset for VSCO2.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://www.openslr.org/resources/1/waves_yesno.tar.gz"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"waves_yesno"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: nn.Module = None,
        folder_in_archive: str = _VSCO2_RELEASE_CONFIGS["release1"]["folder_in_archive"],
        url: str = _VSCO2_RELEASE_CONFIGS["release1"]["url"],
        download: bool = False
    ) -> None:
        self.transform = transform
        self._parse_filesystem(root, url, folder_in_archive, download)

    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
        root = Path(root)
        archive = os.path.basename(url)
        archive = root / archive
        self._path = root / folder_in_archive
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    # checksum = _VSC02_RELEASE_CONFIGS["release1"]["checksum"]
                    # download_url(url, root, hash_value=checksum)
                    download_url(url, root)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        file_paths = Path(self._path).rglob("*.wav")
        self._walker = sorted(file_paths)

    def _load_item(self, file_path: Path):
        labels = file_path.parent.as_posix().split(sep='/')
        # TODO: add key if it exists
        waveform, sample_rate = torchaudio.load(file_path.as_posix())
        audio = self.transform(waveform) if self.transform is not None else waveform
        return (audio, sample_rate), labels

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        file_path = self._walker[n]
        item = self._load_item(file_path, self._path)
        return item

    def __len__(self) -> int:
        return len(self._walker)

class YESNOPacked(Dataset):
    """Same as YESNO but 
    __getitem__ returns sampler rate packed with audio data in a tuple.
    Also interfaced the same as VSCO2 for compatibility.

    Args:
        Dataset ([type]): [description]
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: nn.Module = None,
        folder_in_archive: str = _YESNO_RELEASE_CONFIGS["release1"]["folder_in_archive"],
        url: str = _YESNO_RELEASE_CONFIGS["release1"]["url"],
        download: bool = False
    ) -> None:
        self.transform = transform
        self.dataset = YESNO(root, url, folder_in_archive, download)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        waveform, sr, labels = self.dataset[n]
        audio = self.transform(waveform) if self.transform is not None else waveform
        return (audio, sr), labels

    def __len__(self) -> int:
        return len(self.dataset)