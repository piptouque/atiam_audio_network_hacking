import torch.nn.functional as func
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import ImageVae
from .metric import  vae_divergence_loss, vae_reconstruction_loss


def nll_loss(output: torch.Tensor, target: torch.Tensor, _: nn.Module = None) -> Variable:
    return func.nll_loss(output, target)

def bce_loss(output: torch.Tensor, target: torch.Tensor, _: nn.Module = None) -> Variable:
    return func.binary_cross_entropy(output, target)

def vae_total_loss(output: torch.Tensor, target: torch.Tensor, model: ImageVae) -> Variable:
    return vae_reconstruction_loss(output, target, model) + vae_divergence_loss(output, target, model)