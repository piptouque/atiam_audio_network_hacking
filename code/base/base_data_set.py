import abc
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()