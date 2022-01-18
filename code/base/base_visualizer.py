import abc
import torch
from torch.autograd import Variable
from utils.writer import TensorboardWriter
from base import BaseModel, BaseDataLoader


class BaseVisualizer:
    """[summary]
    """
    def __init__(self) -> None:
        # setup visualization writer instance                
        self.writer = TensorboardWriter()
        self._vis_cfg = None
        self._cfg = None

    def set_up(self, log_dir: str, cfg: dict) -> bool:
        self._cfg = cfg
        self._vis_cfg = self._cfg['visualization']
        succeeded = self.writer.set_up(log_dir, self._vis_cfg['tensorboard'])
        return succeeded

    @abc.abstractmethod
    def log_batch_train(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        """Called after computing training on a single batch.

        Args:
            epoch (int): [description]
            batch_idx (int): [description]
            data (torch.Tensor): [description]
            output (torch.Tensor): [description]
            label (torch.Tensor): [description]
            loss (Variable): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_epoch_train(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        """Called after computing training on a single epoch.

        Args:
            epoch (int): [description]
            batch_idx (int): [description]
            data (torch.Tensor): [description]
            output (torch.Tensor): [description]
            label (torch.Tensor): [description]
            loss (Variable): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_batch_valid(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        """Called after performing validation on a single batch.

        Args:
            epoch (int): [description]
            batch_idx (int): [description]
            data (torch.Tensor): [description]
            output (torch.Tensor): [description]
            label (torch.Tensor): [description]
            loss (Variable): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_epoch_valid(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        """Called after performing validation on a single epoch.

        Args:
            epoch (int): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError