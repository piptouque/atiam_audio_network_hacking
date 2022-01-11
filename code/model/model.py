import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.autograd import Variable

from typing import Tuple, Callable, Union, Any
import abc
from utils import get_output_shape
from base import BaseModel


class RandomSampler(BaseModel):
    def __init__(self, input_size: int, output_size: int, prior_distrib: torch.distributions.Distribution) -> None:
        super().__init__()
        self._input_size = int(input_size)
        self._output_size = int(output_size)
        self._prior_distrib = prior_distrib
        self.input_last = None

        def hook(module: RandomSampler, args: Tuple[torch.Tensor]) -> None:
            self.input_last = args[0]
        self.register_forward_pre_hook(hook)

    @abc.abstractmethod
    def log_likelihood(self, target: torch.Tensor, input_last: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def kl_divergence(self, target: torch.Tensor, input_last: torch.Tensor) -> torch.Tensor:
        return func.kl_div(self.log_likelihood(input_last), self._prior_distrib.log_prob(target), log_target=True)


class GaussianRandomSampler(RandomSampler):
    r"""
    Actually only works with a Gaussian distribution
    see:   Kingma, Diederik P., et Max Welling. « Auto-Encoding Variational Bayes ». ArXiv:1312.6114 [Cs, Stat], 1 mai 2014. http://arxiv.org/abs/1312.6114.
    """

    def __init__(self, input_size: int, output_size: int, fixed_var: Union[None, torch.Tensor] = None) -> None:
        super().__init__(input_size, output_size, torch.distributions.Normal(0, 1))
        self._l_1 = nn.Sequential(
            nn.Linear(self._input_size, self._input_size),
            nn.Tanh(),
        )
        self._l_mean = nn.Linear(self._input_size, self._output_size)
        if fixed_var is None:
            self._l_logscale = nn.Linear(
                self._input_size, self._output_size)
        else:
            self._l_logscale = lambda _: torch.log(
                torch.tensor(fixed_var))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z_mean, z_scale = self._l_moments(y)
        z = z_mean + z_scale * self._prior_distrib.sample()
        return z

    def _l_moments(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_h = self._l_1(y)
        return self._l_mean(y_h), torch.exp(self._l_logscale(y_h))

    def kl_divergence(self, _x: torch.Tensor, input_last: torch.Tensor) -> torch.Tensor:
        z_mean, z_scale = self._l_moments(input_last)
        return 0.5 + torch.log(z_scale) - (z_scale ** 2 + z_mean ** 2)

    def log_likelihood(self, output: torch.Tensor, input_last: torch.Tensor) -> torch.Tensor:
        """ adapted from: https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.log_prob
        Args:
            x (torch.Tensor): [description]
            input_last (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        out_mean, out_scale = self._l_moments(input_last)
        eps = 10e-3
        return - ((output - out_mean) ** 2 / (2 * (out_scale + eps) ** 2) + torch.log(out_scale + eps) + np.log(np.sqrt(2*np.pi)))


class BernoulliRandomSampler(RandomSampler):
    r"""
    The prior and posterior are still gaussian.
    Used as a decoder when the input data (and so, output data) is binary.
    """

    def __init__(self, input_dim: Tuple[int, int, int], output_dim: Tuple[int, int, int]) -> None:
        super().__init__(input_dim, output_dim, torch.distributions.Bernoulli(probs=0.5))
        self._l_probs = nn.Sequential(
            nn.Linear(self._input_size, self._output_size),
            nn.Tanh(),
            nn.Linear(self._output_size, self._output_size),
            nn.Sigmoid()
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        probs = self._l_probs(y)
        z = torch.bernoulli(probs)
        return z

    def log_likelihood(self, output: torch.Tensor, input_last: torch.Tensor) -> torch.Tensor:
        # adapted from: https://pytorch.org/docs/stable/_modules/torch/distributions/bernoulli.html#Bernoulli.log_prob
        probs = self._l_probs(input_last)
        log_probs = - \
            func.binary_cross_entropy(probs, output, reduction='none')
        return log_probs

# references:
# https://keras.io/examples/generative/vae/
# https://avandekleut.github.io/vae/


class VariationalEncoder(BaseModel):
    def __init__(self, input_dim: Tuple[int, int, int], latent_size: int, conv_cfg: dict, sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self._l_1 = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, **conv_cfg),
            nn.ReLU(),
            nn.Conv2d(32, 64, **conv_cfg),
            nn.ReLU()
        )

        conv_out_shape = get_output_shape(self._l_1, (1,) + tuple(input_dim))
        conv_out_dim = np.array(conv_out_shape)[1:]
        conv_size = np.prod(np.array(conv_out_dim))

        self._l_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_size, 16),
            nn.ReLU()
        )

        self.sampler = sampler_fac(16, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_prob = self._l_1(x)
        z_prob = self._l_2(z_prob)
        z = self.sampler(z_prob)
        return z


class VariationalDecoder(BaseModel):
    """[summary]
    Note that it does not return the estimated $p(x)$, but $\hat{x}$ directly.
    Args:
        BaseModel ([type]): [description]
    """

    def __init__(self, output_dim: Tuple[int, int, int], latent_size: int, conv_cfg: dict, sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self._output_dim = output_dim
        self._input_conv_dim = np.array(output_dim)
        #
        # infer reduced height and width
        stride = conv_cfg['stride']
        self._input_conv_dim[-2:] = self._input_conv_dim[-2:]//stride**2
        # channel
        self._input_conv_dim[0] = self._input_conv_dim[0] * 64
        #
        self._input_conv_dim = tuple(self._input_conv_dim)
        conv_size = np.prod(np.array(self._input_conv_dim))
        #
        self._l_1 = nn.Sequential(
            nn.Linear(latent_size, conv_size),
            nn.ReLU()
        )
        self._l_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, **conv_cfg),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_dim[0], **conv_cfg),
            nn.ReLU(),
            nn.Flatten()
        )

        l_2_out_shape = get_output_shape(
            self._l_2, (1,) + tuple(self._input_conv_dim))
        l_2_out_dim = np.array(l_2_out_shape)[1:]
        l_2_size = np.prod(np.array(l_2_out_dim))
        #
        output_size = np.prod(np.array(self._output_dim))
        self.sampler = sampler_fac(l_2_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_prob = self._l_1(z)
        x_prob = torch.reshape(
            x_prob, (z.shape[0],) + tuple(self._input_conv_dim))
        x_prob = self._l_2(x_prob)
        x_hat = self.sampler(x_prob)
        x_hat = torch.reshape(
            x_hat, (z.shape[0],) + tuple(self._output_dim))
        return x_hat


class Vae(BaseModel):
    def __init__(self, input_dim: Tuple[int, int, int], latent_size: int, conv_cfg: dict, e_sampler_fac: Callable[[Any], RandomSampler], d_sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self.encoder = VariationalEncoder(
            input_dim, latent_size, conv_cfg, sampler_fac=e_sampler_fac)
        self.decoder = VariationalDecoder(
            input_dim, latent_size, conv_cfg, sampler_fac=d_sampler_fac)
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class BetaScheduler:
    """
    Inspired by: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR
    """

    def __init__(self, init_val: float, step_size: int, gamma: float, bounds: Tuple[float, float], last_epoch: int = -1) -> None:
        self.last_epoch = last_epoch
        self.step_size = int(step_size)
        self.gamma = gamma
        self.init_val = init_val
        self._last_val = self.init_val
        self.bounds = bounds

    def get_value(self) -> float:
        if self.last_epoch < 0:
            return self.init_val
        else:
            val = self.init_val * \
                (1 + self.gamma * (self.last_epoch // self.step_size))
            return min(self.bounds[1], max(self.bounds[0], val))

    def step(self, epoch: int) -> None:
        self.last_epoch = int(epoch)


class BetaVae(Vae):
    def __init__(self, input_dim: Tuple[int, int, int], latent_size: int, conv_cfg: dict, e_sampler_fac: Callable[[Any], RandomSampler], d_sampler_fac: Callable[[Any], RandomSampler], beta_scheduler: BetaScheduler) -> None:
        super().__init__(input_dim, latent_size, conv_cfg, e_sampler_fac, d_sampler_fac)
        self.beta_scheduler = beta_scheduler

    @ property
    def beta(self) -> float:
        return self.beta_scheduler.get_value()


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, training=self.training)
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)
