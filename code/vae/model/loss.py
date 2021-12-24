import torch.nn.functional as func
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import MnistVae


def nll_loss(output: torch.Tensor, target: torch.Tensor, model_: nn.Module) -> Variable:
    return func.nll_loss(output, target)

def bce_loss(output: torch.Tensor, target: torch.Tensor, model_: nn.Module) -> Variable:
    return func.binary_cross_entropy(output, target)

def kl_loss(output: torch.Tensor, target: torch.Tensor, model_: nn.Module) -> Variable:
    return func.kl_div(output, target) 

def vae_loss(output: torch.Tensor, target: torch.Tensor, model: MnistVae) -> Variable:
    #self.aux_loss = (z_var ** 2 + z_mean ** 2 - torch.log(z_var) - 0.5).sum()
    z_var, z_mean = model.encoder.sampler.buffer_mean, model.encoder.sampler.buffer_var
    sampler_loss = (z_var ** 2 + z_mean ** 2 - torch.log(z_var) - 0.5).sum()
    return bce_loss(output, target, model) + sampler_loss