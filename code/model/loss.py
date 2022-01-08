import numpy as np
import torch.nn.functional as func
import torch
import torch.nn as nn
from torch.autograd import Variable
from .model import Vae, BetaVae
from .metric import vae_divergence_loss, vae_reconstruction_loss


def nll_loss(output: torch.Tensor, target: torch.Tensor, _: nn.Module = None) -> Variable:
    return func.nll_loss(output, target)


def bce_loss(output: torch.Tensor, target: torch.Tensor, _: nn.Module = None) -> Variable:
    return func.binary_cross_entropy(output, target)


def vae_total_loss(output: torch.Tensor, target: torch.Tensor, model: Vae) -> Variable:
    return vae_reconstruction_loss(output, target, model) + vae_divergence_loss(output, target, model)


def beta_vae_total_loss(output: torch.Tensor, target: torch.Tensor, model: BetaVae) -> Variable:
    """
    Using the normalised $\beta$ here. 
    See Higgins et al. for details

    Args:
        output (torch.Tensor): [description]
        target (torch.Tensor): [description]
        model (Vae): [description]
        beta (float): [description]

    Returns:
        Variable: [description]
    """
    input_size = np.product(np.array(output.shape)[1:])
    latent_size = model.latent_size
    beta_norm = model.beta * latent_size / input_size
    return vae_reconstruction_loss(output, target, model) + beta_norm * vae_divergence_loss(output, target, model)
