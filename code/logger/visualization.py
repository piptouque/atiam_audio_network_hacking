import torch
import torchvision.utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
# tofix: why do I need to use base.base_visualizer instead of just base?
from base.base_visualizer import BaseVisualizer
from base import BaseDataLoader, BaseModel
from model import ImageVae


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

class ImageVaeVisualizer(UnsupervisedVisualizer):
    """[summary]

    Args:
        UnsupervisedVisualizer ([type]): [description]
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def log_batch_train(self, model: ImageVae, epoch: int, batch_idx: int, data: torch.Tensor, output: torch.Tensor, label: torch.Tensor, loss: Variable) -> None:
        super().log_batch_train(model, epoch, batch_idx, data, output, label, loss)
        assert model.latent_size == 2, "NOoooooooO"
        if self.vis_cfg['plot_sampled_latent']:
            sampled_latent_dim = self.vis_cfg['sampled_latent_dim']
            x_hat = self._sample_latent(model, sampled_latent_dim)
            self.writer.add_image('sampled_latent', torchvision.utils.make_grid(
                x_hat.cpu(), nrow=sampled_latent_dim[0], normalize=True))

    def log_epoch_train(self, model: ImageVae, epoch: int, data_loader: BaseDataLoader) -> None:
        super().log_epoch_train(model, epoch, data_loader)
        if self.vis_cfg['plot_clusters_latent']:
            clusters_latent_size = self.vis_cfg['clusters_latent_size']
            fig = plt.figure()
            fig, ax = plt.subplots()

            x, y, c = self._cluster_latent(model, data_loader, clusters_latent_size)
            # .. do other stuff
            # plot to ax3
            coll = ax.scatter(x, y, c=c, cmap='tab10')
            fig.colorbar(coll)
            self.writer.add_figure('clusters_latent', fig)

    def _sample_latent(self, model: ImageVae, dim: Tuple[int, int]) -> torch.Tensor:
        """Get a tensor of images
        linearly spaced coordinates corresponding to the 
        classes in the latent space.
        Args:
            dim (Tuple[int, int]): [description]

        Returns:
            torch.Tensor: [description]
        """
        assert len(dim) == model.latent_size
        with torch.no_grad():
            z_1 = np.linspace(-1, 1, dim[0])
            z_2 = np.linspace(-1, 1, dim[1])[::-1]
            zz_1, zz_2 = np.meshgrid(z_1, z_2)
            z = torch.cat((torch.tensor(zz_1[..., np.newaxis], dtype=torch.float), torch.tensor(zz_2[..., np.newaxis], dtype=torch.float)), -1)
            z = torch.flatten(z, start_dim=0, end_dim=-2)
            x_hat = model.decoder(z)
            return x_hat

    def _cluster_latent(self, model: ImageVae, data_loader: BaseDataLoader, nb_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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