
from typing import List, Callable
from typing import Callable
from base.base_visualizer import BaseVisualizer
from utils.writer import TensorboardWriter
from numpy import inf
from abc import abstractmethod
from torch.autograd import Variable
import torch.nn as nn
import torch

LossCriterion = Callable[[torch.Tensor, torch.Tensor, nn.Module], Variable]


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion: LossCriterion, metric_ftns: List[Callable], optimizer, visualizer: BaseVisualizer, logger, config: dict) -> None:
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.logger = logger
        cfg_trainer = config['training']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        self.visualizer = visualizer
        succeeded = self.visualizer.set_up(config.log_dir, config.config)
        if not succeeded:
            message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the config file."
            logger.warning(message)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.visualizer = visualizer
        self.config = config
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> None:
        """Training logic for a single epoch.

        Args:
            epoch (int): Epoch iteration number

        """
        raise NotImplementedError

    @abstractmethod
    def _log_batch(self, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, target: torch.Tensor, loss: Variable) -> None:
        """Optional logging after each processed batch.

        Args:
            epoch (int): Epoch iteration number
            batch_idx (int): Batch number
            data (torch.Tensor): Input data
            output (torch.Tensor): Output of model
            target (torch.Tensor): Target data
            loss (Variable): Model loss
        """
        pass

    def train(self) -> None:
        """Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode ==
                                'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _progress(self, batch_idx: int) -> str:
        """[summary]

        Args:
            batch_idx (int): Batch number

        Returns:
            str: Progress logged as a string
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """Saves latest weights.
        Args:
            epoch (int): Epoch iteration number.
            save_best (bool, optional): If True, renames the checkpoint to 'model_best.pth'. Defaults to False.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Resume from saved checkpoints

        Args:
            resume_path (str): param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
