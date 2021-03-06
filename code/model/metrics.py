import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

from .models import Vae


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, _model: nn.Module) -> float:
    pred = torch.argmax(output, dim=-1)
    # assert pred.shape[0] == len(target)
    correct = 0
    correct += torch.sum(pred == target).item()
    return correct / len(target)


@torch.no_grad()
def top_k_acc(output: torch.Tensor, target: torch.Tensor, _model: nn.Module, k: int = 3) -> float:
    pred = torch.topk(output, k=k, dim=-1)[1]
    # assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def vae_reconstruction_loss(_output: torch.Tensor, target: torch.Tensor, model: Vae) -> Variable:
    # dim = tuple(np.arange(-target.ndim + 1, 0))
    # first sum over each sample in the batch
    # then average over the batch.
    target_flat = torch.flatten(target, start_dim=1)
    loss = - model.decoder.sampler.log_likelihood(
        target_flat, model.decoder.sampler.input_last)
    dim = tuple(np.arange(1, loss.dim()))
    loss = loss.sum(dim=dim).mean()
    return loss


def vae_divergence_loss(_output: torch.Tensor, target: torch.Tensor, model: Vae) -> Variable:
    # dim = tuple(np.arange(1, target.ndim))
    loss = - model.encoder.sampler.kl_divergence(
        target, model.encoder.sampler.input_last)
    dim = tuple(np.arange(1, loss.dim()))
    loss = loss.sum(dim=dim).mean()
    return loss
