from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BinaryMnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, batch_size: int, shuffle=True, validation_split=0.0, num_workers=1, training=True) -> None:
        self.data_dir = data_dir
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.round()) # to black and white
        ])
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
