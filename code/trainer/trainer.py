import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from base import BaseTrainer, LossCriterion
from utils import inf_loop, MetricTracker

import matplotlib.pyplot as plt

from model import ImageVae
from typing import Tuple, List

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion: LossCriterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _choose_target(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return label

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)

            target = self._choose_target(data, label)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss =  self.criterion(output, target, self.model)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(output, target, self.model))

            if batch_idx % self.log_step == 0:
                self._log_batch(epoch, batch_idx, data, output, label, loss)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.valid_data_loader):
                data, label = data.to(self.device), label.to(self.device)
                target = self._choose_target(data, label)

                output = self.model(data)
                loss = self.criterion(output, target, self.model)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target, self.model))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _log_batch(self, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable):
        self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            epoch,
            self._progress(batch_idx),
            loss.item()))
        self.writer.add_image('input', make_grid(
            data.cpu(), nrow=8, normalize=True))


class UnsupervisedTrainer(Trainer):
    """
    Trainer class
    """
    def _choose_target(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return data

    def _log_batch(self, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable):
        super()._log_batch(epoch, batch_idx, data, output, label, loss)
        self.writer.add_image('output', make_grid(
            output.cpu(), nrow=8, normalize=True))

class ImageVaeTrainer(UnsupervisedTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.model, ImageVae), "Nooo"
        assert self.model.latent_size == 2, "NOoooooooO"

    def _log_batch(self, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable):
        super()._log_batch(epoch, batch_idx, data, output, label, loss)
        vis_cfg = self.config['training']['visualization']
        if vis_cfg['plot_sampled_latent']:
            sample_dim = tuple(vis_cfg['sampled_latent_dim'])
            x_hat = self._sample_latent(sample_dim)
            self.writer.add_image('sampled_latent', make_grid(
                x_hat.cpu(), nrow=sample_dim[0], normalize=True))
        if vis_cfg['plot_clusters_latent']:
            fig = plt.figure()
            fig, ax = plt.subplots()

            x, y, c = self._cluster_latent(data, label)
            # .. do other stuff
            # plot to ax3
            ax.scatter(x, y, c=c, cmap='tab10')
            self.writer.add_figure('clusters_latent', fig)



    def _sample_latent(self, dim: Tuple[int, int]) -> torch.Tensor:
        """Get a tensor of images
        linearly spaced coordinates corresponding to the 
        classes in the latent space.
        Args:
            dim (Tuple[int, int]): [description]

        Returns:
            torch.Tensor: [description]
        """
        assert len(dim) == self.model.latent_size
        decoder = self.model.decoder
        z_1 = np.linspace(-1, 1, dim[0])
        z_2 = np.linspace(-1, 1, dim[1])[::-1]
        zz_1, zz_2 = np.meshgrid(z_1, z_2)
        z = torch.cat((torch.tensor(zz_1[..., np.newaxis], dtype=torch.float), torch.tensor(zz_2[..., np.newaxis], dtype=torch.float)), -1)
        z = torch.flatten(z, start_dim=0, end_dim=-2)
        x_hat = self.model.decoder(z)
        return x_hat

    def _cluster_latent(self, data: torch.Tensor, label: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            z = self.model.encoder(data).cpu().detach().numpy()
            x = z[..., 0]
            y = z[..., 1]
            c = label.cpu().numpy()
            # .. do other stuff
            # plot to ax3
            return x, y, c
