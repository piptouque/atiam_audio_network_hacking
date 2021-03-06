import numpy as np
import torch
from torch.autograd import Variable
from base import BaseTrainer, LossCriterion, DataLoader
from utils import inf_loop, MetricTracker

from model import Vae, BetaVae
from typing import Tuple, List


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion: LossCriterion, metric_ftns, optimizer, visualizer, logger, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns,
                         optimizer, visualizer, logger, config)
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
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.visualizer.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.visualizer.writer)

    def _choose_target(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return label

    def _forward_model(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        if self.beta_scheduler
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(self.device), label.to(self.device)

            target = self._choose_target(data, label)

            self.optimizer.zero_grad()
            output = self._forward_model(data)
            loss = self.criterion(output, target, self.model)
            loss.backward()
            self.optimizer.step()

            self.visualizer.writer.set_step(
                (epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(output, target, self.model))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.visualizer.log_batch_train(
                    self.model, epoch, batch_idx, data, output, label, loss)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        self.visualizer.log_epoch_train(self.model, epoch, self.data_loader)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        self.lr_scheduler.step()

        self._post_epoch(epoch)

        return log

    def _post_epoch(self, epoch: int) -> None:
        pass

    def _valid_epoch(self, epoch: int):
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

                self.visualizer.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target, self.model))
                self.visualizer.log_batch_valid(
                    self.model, epoch, batch_idx, data, output, label, loss)

        self.visualizer.log_epoch_valid(
            self.model, epoch, self.valid_data_loader)

        return self.valid_metrics.result()


class UnsupervisedTrainer(Trainer):
    """
    Trainer class
    """

    def _choose_target(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return data


class BetaVaeTrainer(UnsupervisedTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, visualizer, logger, config, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None) -> None:
        super(BetaVaeTrainer, self).__init__(model, criterion, metric_ftns, optimizer, visualizer, logger, config,
                                             device, data_loader, valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, len_epoch=len_epoch)
        assert isinstance(self.model, BetaVae)

    def _post_epoch(self, epoch: int) -> None:
        self.model.beta_scheduler.step(epoch)


class AdversarialTrainer:
    """
    Trainer class for Adversarial training
    of a generator model with a discriminator model.
    For now, does not actually train the generator.
    """

    def __init__(self, gen_trainer: Trainer, dis_trainer: BaseTrainer):
        self.gen_trainer = gen_trainer
        self.gen_data_loader = self.gen_trainer.data_loader
        self.dis_trainer = dis_trainer
        self.dis_data_loader = self.dis_trainer.data_loader

        self.start_epoch = 1
        self.epochs = self.gen_trainer.epochs

        self.logger = self.gen_trainer.logger

        # TODO: set right training loss for generator!
        # self.gen_trainer.criterion =

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        switch_epochs = 22
        epoch_gen = 0
        epoch_dis = 0

        for epoch_global in range(self.start_epoch, self.epochs + 1):
            # training_gen = not training_gen
            # training_trainer = self.gen_trainer if training_gen else self.dis_trainer
            training_trainer = self.dis_trainer
            training_gen = False
            for epoch_switch in range(1, switch_epochs + 1):
                epoch = epoch_global * epoch_switch
                result = training_trainer._train_epoch(epoch)

                # save logged informations into log dict
                log = {
                    'phase': 'Generation' if training_gen else 'Classification',
                    'epoch': epoch
                }
                log.update(result)

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if epoch // 2 % training_trainer.save_period == 0:
                    training_trainer._save_checkpoint(
                        epoch // 2, save_best=best)
                if training_gen:
                    epoch_gen += 1
                else:
                    epoch_dis += 1
