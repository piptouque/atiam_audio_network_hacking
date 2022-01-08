
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import Dataset
from torchaudio.datasets import YESNO
from torchaudio.datasets.utils import download_url, extract_archive
from torchaudio.datasets.yesno import _RELEASE_CONFIGS as _YESNO_RELEASE_CONFIGS

_VSCO2_RELEASE_CONFIGS = {
    "hardcore": {
        "folder_in_archive": "VSCO2_hardcore",
        "archive_name": "50OrchestralSamples.zip",
        "url": "https://uc83fded2dd0e02e3c1b77154085.dl.dropboxusercontent.com/zip_download_get/BAxqx1PzWHqR-1JbJe10zyPjiur_6t1HXOZa7pQpFbybBIMNCMgtfdd9xjjmdpi4Ec3vkjH-NocLYmcq4i3SxqIpgaU1Amri6p53ixbY3UPoPA?_download_id=228461718975105252695810256925787335041281438952017275431986463752",
        "checksum": None
    },
    "partial": {
        "folder_in_archive": "VSCO2_partial",
        "archive_name": "256OrchestralSamples.zip",
        "url": "https://uc5ee82ae5da88cfddddd9b91de7.dl.dropboxusercontent.com/zip_download_get/BAzIicx7d8Ew39KmOaP__sjxd2ik_yzuPGaYVml_cq1Ie9G7uGmz8L0HcFCzzTOOEfokwJ44y9KLDjJQElfbw-Y4YrTd7WCejw9ZZGdkHvbpmQ?_download_id=144379115872197751745977629871310968943451386519978745509876583",
        "checksum": None
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
        cfg = _VSCO2_RELEASE_CONFIGS["hardcore"],
        download: bool = False
    ) -> None:
        self.transform = transform
        self._parse_filesystem(root, cfg, download)

    def _parse_filesystem(self, root: str, cfg, download: bool) -> None:
        root = Path(root)
        folder_in_archive, url, checksum = cfg["folder_in_archive"], cfg["url"], cfg["checksum"]
        archive_name = cfg["archive_name"]
        archive = root / archive_name
        self._path = root / folder_in_archive
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive, self._path)

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
        audio = self.transform(
            waveform) if self.transform is not None else waveform
        return audio, sample_rate, labels

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        file_path = self._walker[n]
        item = self._load_item(file_path)
        return item

    def __len__(self) -> int:
        return len(self._walker)


class YESNOPacked(Dataset):
    """Same as YESNO but 
    Interfaced the same as VSCO2 for compatibility.

    Args:
        Dataset ([type]): [description]
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: nn.Module = None,
        cfg = _YESNO_RELEASE_CONFIGS["release1"],
        download: bool = False
    ) -> None:
        self.transform = transform
        self.dataset = YESNO(root, cfg["url"], cfg["folder_in_archive"], download)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, List[int]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, List[int]): ``(waveform, sample_rate, labels)``
        """
        waveform, sr, labels = self.dataset[n]
        audio = self.transform(
            waveform) if self.transform is not None else waveform
        return audio, sr, labels

    def __len__(self) -> int:
        return len(self.dataset)
