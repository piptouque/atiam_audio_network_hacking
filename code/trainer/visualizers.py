import pathlib
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
from data_loader.data_loaders import FashionMnistDataLoader
from model import Vae, BetaVae


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
    """Visaliser for unsupervised training. Adds output of the generative model.

    Args:
        BaseVisualizer ([type]): [description]
    """

    def log_batch_train(self, model: BaseModel, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable):
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        self.writer.add_image('output', torchvision.utils.make_grid(
            output.cpu(), nrow=8, normalize=True))


class VaeVisualizer(UnsupervisedVisualizer):
    """Visualiser for VAEs. Adds sampling and clustering of latent space.

    Args:
        UnsupervisedVisualizer ([type]): [description]
    """

    def set_up(self, log_dir: str, cfg: dict) -> bool:
        succeeded = super(VaeVisualizer, self).set_up(log_dir, cfg)
        data_dir = cfg['data_loader']['kwargs']['data_dir']
        self._symbol_batch_size = self._vis_cfg['exploration_latent']['nb_samples']
        self._symbols_loader = FashionMnistDataLoader(
            data_dir, self._symbol_batch_size, training=False)

        # create a directory for the SVG versions of the latent space plots.
        self._cluster_fig_path = pathlib.Path(self.writer._log_dir) / \
            'clusters_latent'
        self._cluster_fig_path.mkdir(parents=True, exist_ok=True)
        return succeeded

    def log_batch_train(self, model: Vae, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        s_cfg = self._vis_cfg['sampled_latent']
        if s_cfg['plot']:
            assert model.latent_size == 2, "NOoooooooO"
            nb_points = s_cfg['nb_points']
            lims = s_cfg['range']
            x_hat = self._sample_latent(model, lims, nb_points)
            self.writer.add_image('sampled_latent', torchvision.utils.make_grid(
                x_hat.cpu(), nrow=nb_points[0], normalize=True))

    def log_epoch_train(self, model: Vae, epoch: int, data_loader: BaseDataLoader) -> None:
        super().log_epoch_train(model, epoch, data_loader)
        cluster_cfg = self._vis_cfg['clusters_latent']
        if cluster_cfg['plot']:
            assert model.latent_size == 2, "NOoooooooO"
            nb_points = cluster_cfg['nb_points']
            fig = plt.figure()
            fig, ax = plt.subplots()

            x, y, c = self._cluster_latent(model, data_loader, nb_points)
            # .. do other stuff
            # plot to ax3
            coll = ax.scatter(x, y, c=c, cmap='tab10')
            fig.colorbar(coll)
            fig.savefig(self._cluster_fig_path / f'{epoch}.svg', format='svg')
            # also save in tensorbord, why not.
            self.writer.add_figure('clusters_latent', fig)
            plt.close(fig)

        destroy_cfg = self._vis_cfg['destroy_output']
        if destroy_cfg['plot']:
            # first remember the uncorrumpted weights
            clean_weights = copy.deepcopy(model.state_dict())
            # corrupt them
            self._weights_reset_rand(model, destroy_cfg['corruption_rate'])
            # plot the output of the corrupted model
            x_destroyed = self._explore_destroyed(model, data_loader)
            self.writer.add_image('destroyed_output', torchvision.utils.make_grid(
                x_destroyed.cpu(), nrow=self._symbol_batch_size, normalize=True))
            # load back the clean weights
            model.load_state_dict(clean_weights)
        exploration_cfg = self._vis_cfg['exploration_latent']
        if exploration_cfg['plot']:
            x_noise = self._explore_noise(
                model, exploration_cfg['nb_samples'], exploration_cfg['nb_iter'], data_loader)
            x_symbol = self._explore_symbols(
                model, exploration_cfg['nb_iter'], data_loader)
            self.writer.add_image('exploration_noise', torchvision.utils.make_grid(
                x_noise.cpu(), nrow=exploration_cfg['nb_samples'], normalize=True))
            self.writer.add_image('exploration_symbols', torchvision.utils.make_grid(
                x_symbol.cpu(), nrow=exploration_cfg['nb_samples'], normalize=True))

    def _weights_reset_rand(self, model: Vae, rate: float):
        for layer in model.children():
            should_reset = np.random.binomial(1, rate)
            if hasattr(layer, 'reset_parameters') and should_reset:
                layer.reset_parameters()

    def _explore_destroyed(self, model: Vae, data_loader: BaseDataLoader) -> torch.Tensor:
        data, _ = next(iter(data_loader))
        x = model(data)
        return x

    def _explore_noise(self, model: Vae, nb_samples: int, nb_iter: int, data_loader: BaseDataLoader) -> torch.Tensor:
        data, _ = next(iter(data_loader))
        shape = (nb_samples,) + data.shape[1:]
        x_i = torch.rand(shape, dtype=data.dtype)
        x = [x_i]
        for i in range(nb_iter-1):
            x_i = model(x_i)
            x.append(x_i)
        x = torch.cat(x, dim=0)
        return x

    def _explore_symbols(self, model: Vae, nb_iter: int, data_loader: BaseDataLoader) -> torch.Tensor:
        x_j, _ = next(iter(self._symbols_loader))
        x = [x_j]
        for j in range(nb_iter-1):
            x_j = model(x_j)
            x.append(x_j)
        x = torch.cat(x, dim=0)
        return x

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
            z = torch.cat((torch.tensor(zz_1[..., np.newaxis], dtype=torch.float), torch.tensor(
                zz_2[..., np.newaxis], dtype=torch.float)), -1)
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


class BetaVaeVisualizer(VaeVisualizer):
    """Visualiser for $\beta$-VAEs. Adds logging of the $\beta$ parameter's variation.

    Args:
        UnsupervisedVisualizer ([type]): [description]
    """

    def log_batch_train(self, model: BetaVae, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        self.writer.add_scalar('beta', model.beta)
