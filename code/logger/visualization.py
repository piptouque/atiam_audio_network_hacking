import torch
import torch.nn as nn
import torchvision.utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy

from typing import Tuple
# tofix: why do I need to use base.base_visualizer instead of just base?
from base.base_visualizer import BaseVisualizer
from base import BaseDataLoader, BaseModel
from data_loader.data_loaders import BinaryFashionMnistDataLoader
from model import Vae


class Visualizer(BaseVisualizer):
    """[summary]

    Args:
        BaseVisualizer ([type]): [description]
    """
    def log_batch_train(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        self.writer.add_image('input', torchvision.utils.make_grid(
            data.cpu(), nrow=8, normalize=True))

    def log_epoch_train(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        pass

    def log_batch_valid(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        self.writer.add_image('input', torchvision.utils.make_grid(
            data.cpu(), nrow=8, normalize=True))

    def log_epoch_valid(self, model: BaseModel, epoch: int, data_loader: BaseDataLoader) -> None:
        # add histogram of model parameters to the tensorboard
        for name, p in model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
class UnsupervisedVisualizer(Visualizer):
    """[summary]

    Args:
        BaseVisualizer ([type]): [description]
    """
    def log_batch_train(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable):
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        self.writer.add_image('output', torchvision.utils.make_grid(
            output.cpu(), nrow=8, normalize=True))

class VaeVisualizer(UnsupervisedVisualizer):
    """[summary]

    Args:
        UnsupervisedVisualizer ([type]): [description]
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = 8
        self.symbols_loader = BinaryFashionMnistDataLoader("./", self.batch_size, training=False)


    def log_batch_train(self, model: Vae, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        s_cfg = self.vis_cfg['sampled_latent']
        if s_cfg['plot']:
            assert model.latent_size == 2, "NOoooooooO"
            nb_points = s_cfg['nb_points']
            lims = s_cfg['range']
            x_hat = self._sample_latent(model, lims, nb_points)
            self.writer.add_image('sampled_latent', torchvision.utils.make_grid(
                x_hat.cpu(), nrow=nb_points[0], normalize=True))


    def log_epoch_train(self, model: Vae, epoch: int, data_loader: BaseDataLoader) -> None:
        super().log_epoch_train(model, epoch, data_loader)
        clean_weights = copy.deepcopy(model.state_dict())
        print(clean_weights.shape)
        c_cfg = self.vis_cfg['clusters_latent']
        if c_cfg['plot']:
            assert model.latent_size == 2, "NOoooooooO"
            nb_points = c_cfg['nb_points']
            fig = plt.figure()
            fig, ax = plt.subplots()

            x, y, c = self._cluster_latent(model, data_loader, nb_points)
            # .. do other stuff
            # plot to ax3
            coll = ax.scatter(x, y, c=c, cmap='tab10')
            fig.colorbar(coll)
            self.writer.add_figure('clusters_latent', fig)
            plt.close(fig)
        self._log_new_digits(model, epoch, data_loader)
        self._log_new_digits_from_symbols(model, epoch, data_loader)


    def weight_reset_rand(self, model: Vae, rate: float):
        for layer in model.children():
            should_reset = np.random.binomial(1, rate)
            if hasattr(layer, 'reset_parameters') and should_reset:
                layer.reset_parameters()
                

    #def _corrupted_model(self, model: Vae, rate: float) -> Vae:
    #   model = model.detach()
    #   corrupted_model = copy.deepcopy(model)
    #   corrupted_model.weight_reset_rand(model, rate)
    #   return corrupted_model


    def _log_new_digits(self, model: Vae, epoch: int, data_loader: BaseDataLoader) -> None:
        data, _ = next(iter(data_loader))
        nb_points = 4
        nb_iter = 10
        shape = (nb_points,) + data.shape[1:]
        x_i = torch.rand(shape, dtype=data.dtype)
        x = [x_i]
        for i in range(nb_iter-1):
            x_i = model(x_i)
            x.append(x_i)
        x = torch.cat(x, dim=0)
        self.writer.add_image('new_digits', torchvision.utils.make_grid(
                x.cpu(), nrow=nb_points, normalize=True))


    def _log_new_digits_from_symbols(self, model: Vae, epoch: int, data_loader: BaseDataLoader) -> None:
        nb_iter = 10
        x = []
        x_j, _ = next(iter(self.symbols_loader))
        x.append(x_j)
        for j in range(nb_iter-1):
            x_j = model(x_j)
            x.append(x_j)
        x = torch.cat(x, dim=0)
        self.writer.add_image('new_digits_from_symbols', torchvision.utils.make_grid(
                        x.cpu(), nrow=self.batch_size, normalize=True))


    def _sample_latent(self, model: Vae, lims: Tuple[Tuple[int, int], Tuple[int, int]], nb_points: Tuple[int, int]) -> torch.Tensor:
        """Get a tensor of images
        linearly spaced coordinates corresponding to the 
        classes in the latent space.
        Args:
            dim (Tuple[int, int]): [description]

        Returns:
            torch.Tensor: [description]
        """
        assert len(nb_points) == model.latent_size
        assert len(lims) == len(nb_points)
        with torch.no_grad():
            z_1 = np.linspace(lims[0][0], lims[0][1], nb_points[0])
            z_2 = np.linspace(lims[1][0], lims[1][1], nb_points[1])[::-1]
            zz_1, zz_2 = np.meshgrid(z_1, z_2)
            z = torch.cat((torch.tensor(zz_1[..., np.newaxis], dtype=torch.float), torch.tensor(zz_2[..., np.newaxis], dtype=torch.float)), -1)
            z = torch.flatten(z, start_dim=0, end_dim=-2)
            x_hat = model.decoder(z)
            return x_hat


    def _cluster_latent(self, model: Vae, data_loader: BaseDataLoader, nb_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = []
        y = []
        c = []
        n = 0
        with torch.no_grad():
            for _, (data, label) in enumerate(data_loader):
                z = model.encoder(data).cpu().detach().numpy()
                x.append(z[..., 0])
                y.append(z[..., 1])
                c.append(label.cpu().numpy())
                n += z.shape[0]
                if n > nb_points:
                    break
        return x, y, c