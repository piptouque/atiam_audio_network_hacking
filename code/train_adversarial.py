import copy
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import logger.visualization as module_visualization
import trainer as module_trainer

from utils import prepare_device
from utils.parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(gen_config, dis_config):
    logger = dis_config.get_logger(
        'train', dis_config['training']['verbosity'])

    device, device_ids = prepare_device(dis_config['n_gpu'])

    # GENERATOR
    # load generative model
    gen_model = gen_config.init_handle('arch', module_arch, 'obj')
    if gen_config['n_gpu'] > 1:
        gen_model = torch.nn.DataParallel(gen_model)
    gen_model = gen_model.to(device)
    gen_model.eval()

    gen_data_loader = gen_config.init_obj('data_loader', module_data)
    gen_valid_data_loader = gen_data_loader.split_validation()

    # get function handles of loss and metrics
    gen_criterion = gen_config.init_ftn('loss', module_loss)
    gen_metric_ftns = [gen_config.init_ftn(['metrics', i], module_metric)
                       for i in range(len(gen_config['metrics']))]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    gen_trainable_params = filter(
        lambda p: p.requires_grad, gen_model.parameters())
    gen_optimizer = gen_config.init_obj(
        'optimizer', torch.optim, gen_trainable_params)
    gen_lr_scheduler = gen_config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, gen_optimizer)

    visualizer = gen_config.init_obj('visualizer', module_visualization)

    gen_trainer = gen_config.init_obj('trainer', module_trainer,
                                      model=gen_model,
                                      criterion=gen_criterion,
                                      metric_ftns=gen_metric_ftns,
                                      optimizer=gen_optimizer,
                                      visualizer=visualizer,
                                      logger=logger,
                                      config=gen_config,
                                      device=device,
                                      data_loader=gen_data_loader,
                                      valid_data_loader=gen_valid_data_loader,
                                      lr_scheduler=gen_lr_scheduler)

    # DISCRIMINATOR
    # setup data_loader instances
    dis_data_loader = module_data.AdversarialDataloader(
        gen_data_loader, gen_model)
    dis_valid_data_loader = dis_data_loader.split_validation()

    # build classifier model (discriminator) architecture, then print to console
    dis_model = dis_config.init_handle('arch', module_arch, 'obj')
    logger.info(dis_model)

    # prepare for (multi-device) GPU training
    dis_model = dis_model.to(device)
    if len(device_ids) > 1:
        dis_model = torch.nn.DataParallel(dis_model, device_ids=device_ids)

    # get function handles of loss and metrics
    dis_criterion = dis_config.init_ftn('loss', module_loss)
    dis_metric_ftns = [dis_config.init_ftn(['metrics', i], module_metric)
                       for i in range(len(dis_config['metrics']))]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    dis_trainable_params = filter(
        lambda p: p.requires_grad, dis_model.parameters())
    dis_optimizer = dis_config.init_obj(
        'optimizer', torch.optim, dis_trainable_params)
    dis_lr_scheduler = dis_config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, dis_optimizer)

    visualizer = dis_config.init_obj('visualizer', module_visualization)

    dis_trainer = dis_config.init_obj('trainer', module_trainer,
                                      model=dis_model,
                                      criterion=dis_criterion,
                                      metric_ftns=dis_metric_ftns,
                                      optimizer=dis_optimizer,
                                      visualizer=visualizer,
                                      logger=logger,
                                      config=dis_config,
                                      device=device,
                                      data_loader=dis_data_loader,
                                      valid_data_loader=dis_valid_data_loader,
                                      lr_scheduler=dis_lr_scheduler)

    trainer = module_trainer.AdversarialTrainer(gen_trainer, dis_trainer)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='VAE hacking')
    args.add_argument('--gen_resume', required=True, type=str,
                      help='path to generator model checkpoint')
    args.add_argument('--dis_config', default=None, type=str,
                      help='path to discriminator  model config file (default: None)')
    args.add_argument('--dis_resume', default=None, type=str,
                      help='path to discriminator model checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--logger', default=None, type=str,
                      help='logger config path (default: None)')

    # FIXME: ignore options.
    options = []
    # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #    CustomArgs(['--lr', '--learning_rate'],
    #                type=float, target='optimizer;args;lr'),
    #     CustomArgs(['--bs', '--batch_size'], type=int,
    #                target='data_loader;args;batch_size')
    # ]
    args = args.parse_args()
    gen_args = copy.deepcopy(args)
    dis_args = copy.deepcopy(args)
    gen_args.resume = args.gen_resume
    dis_args.config = args.dis_config
    dis_args.resume = args.dis_resume

    gen_config = ConfigParser.from_args(gen_args, options)
    dis_config = ConfigParser.from_args(dis_args, options)
    main(gen_config, dis_config)
