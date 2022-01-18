import abc
import torch
from torch.autograd import Variable
from utils.writer import TensorboardWriter
from base import BaseModel, BaseDataLoader


class BaseVisualizer:
    """Base class for visualisers.
    Holds a reference to the writer and uses it to add data during training.
    """

    def __init__(self) -> None:
        # setup visualization writer instance
        self.writer = TensorboardWriter()
        self._vis_cfg = None
        self._cfg = None

    def set_up(self, log_dir: str, cfg: dict) -> bool:
        """Sets ups up the visualiser instance.

        Args:
            log_dir (str): path to logging directory
            cfg (dict): top config

        Returns:
            bool: Success
        """
        self._vis_cfg = cfg['visualization']
        succeeded = self.writer.set_up(log_dir, self._vis_cfg['tensorboard'])
        return succeeded

    @abc.abstractmethod
    def log_batch_train(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        """Logs info after each batch during training.

        Args:
            epoch (int): Epoch iteration number
            batch_idx (int): Batch number
            data (torch.Tensor): Input data
            output (torch.Tensor): Output data
            label (torch.Tensor): Input label
            loss (Variable): Model loss
        """
        pass

    @abc.abstractmethod
    def log_epoch_train(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        """Logs info after each epoch during training.

        Args:
            epoch (int): Epoch iteration number
            data (torch.Tensor): Input data
            data_loader (BaseDataLoader): training data loader

        """
        pass

    @abc.abstractmethod
    def log_batch_valid(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        """Logs info after each batch during validation.

        Args:
            epoch (int): Epoch iteration number
            batch_idx (int): Batch number
            data (torch.Tensor): Input data
            output (torch.Tensor): Output data
            label (torch.Tensor): Input label
            loss (Variable): Model loss
        """
        pass

    @abc.abstractmethod
    def log_epoch_valid(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        """Logs info after each epoch during validation.

        Args:
            epoch (int): Epoch iteration number
            data (torch.Tensor): Input data
            data_loader (BaseDataLoader): training data loader

        """
        pass
