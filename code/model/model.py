import torch
import torch.nn as nn
import torch.nn.functional as func
from base import BaseModel
from utils import get_output_shape
import numpy as np
from torch.autograd import Variable

from typing import Tuple, Callable, Union, Any
import abc

vae_config = {
    'conv_config': {
        'kernel_size': 3,
        'stride': 2,
        #'padding': 'same'
    },
    'deconv_config': {
        'kernel_size': 3,
        'stride': 2,
        #'output_padding': ''
    }
}

# Important:
# "same" padding currently requires 'stride'==1 in PyTorch
# see: https://github.com/pytorch/pytorch/issues/67551
# So using this work-around to model TensorFlow behaviour
# from: https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
class ConvUtil:
    @staticmethod
    def conv_padding_same(i: int, k: int, s: int, d: int) -> int:
        return np.max((np.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0).astype(int)
    def deconv_padding_same(i: int, k: int, s: int, d: int, p_o: int) -> int:
        pad = np.max((k - 1) * d + 1 + p_o - s, 0).astype(int)
        return pad

class Conv2dSame(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = ConvUtil.conv_padding_same(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = ConvUtil.conv_padding_same(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = func.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return func.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

class ConvTranspose2dSame(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_null = tuple(self.dilation*(np.array(self.kernel_size, dtype=int) - 1)//2)
        pad_h = ConvUtil.deconv_padding_same(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0], p_o=self.output_padding[0])
        pad_w = ConvUtil.deconv_padding_same(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1], p_o=self.output_padding[1])

        x_conv = func.conv_transpose2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=pad_null,
            dilation=self.dilation,
            groups=self.groups,
            output_padding=(pad_h, pad_w)
        )
        return x_conv


class RandomSampler(BaseModel):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: Tuple[int, int, int], prior_distrib: torch.distributions.Distribution) -> None:
        super().__init__()
        self._input_dim = tuple(input_dim)
        self._output_dim = tuple(output_dim)
        self._prior_distrib = prior_distrib
        self._input_last  = None
        def hook(module: RandomSampler, args: Tuple[torch.Tensor]) -> None:
            self._input_last = args[0]
        self.register_forward_pre_hook(hook)
        
    @abc.abstractmethod
    def log_likelihood(self, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def kl_divergence(self, target: torch.Tensor) -> torch.Tensor:
        return func.kl_div(self.log_likelihood(target), self._prior_distrib.log_prob(target), log_target=True)


class GaussianRandomSampler(RandomSampler):
    r"""
    Actually only works with a Gaussian distribution
    see:   Kingma, Diederik P., et Max Welling. « Auto-Encoding Variational Bayes ». ArXiv:1312.6114 [Cs, Stat], 1 mai 2014. http://arxiv.org/abs/1312.6114.
    """
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: Tuple[int, int, int], fixed_var: Union[None, torch.Tensor] = None) -> None:
        super().__init__(input_dim, output_dim, torch.distributions.Normal(0, 1))
        self._l_mean = nn.Linear(self._input_dim[-1], self._output_dim[-1])
        if fixed_var is None: 
            self._l_logscale = nn.Linear(self._input_dim[-1], self._output_dim[-1])
        else:
            self._l_logscale = lambda _: torch.log(fixed_var)

        self.buffer_mean = None

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z_mean, z_scale = self._l_moments(y)
        z = z_mean + z_scale * self._prior_distrib.sample()
        return z
    def _l_moments(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._l_mean(y), torch.exp(self._l_logscale(y))

    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        z_mean, z_scale = self._l_moments(self._input_last)
        return 0.5 + torch.log(z_scale) - (z_scale ** 2 + z_mean ** 2)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        # adapted from: https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.log_prob
        z_mean, z_scale = self._l_moments(self._input_last)
        return  - ((x - z_mean) ** 2 / (2 * z_scale **2) + torch.log(z_scale) + np.log(np.sqrt(2*np.pi)))


class BernoulliRandomSampler(RandomSampler):
    r"""    
    The prior and posterior are still gaussian.
    Used as a decoder when the input data (and so, output data) is binary.
    """
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: Tuple[int, int, int]) -> None:
        super().__init__(input_dim, output_dim, torch.distributions.Bernoulli(probs=0.5))
        self._l_probs = nn.Sequential(
            nn.Linear(self._input_dim[-1], self._output_dim[-1]),
            nn.Tanh(),
            nn.Linear(self._output_dim[-1], self._output_dim[-1]),
            nn.Sigmoid()
        )
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        probs = self._l_probs(y)
        z = torch.bernoulli(probs)
        return z

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        # adapted from: https://pytorch.org/docs/stable/_modules/torch/distributions/bernoulli.html#Bernoulli.log_prob
        probs = self._l_probs(self._input_last)
        m = (x.min(), x.max(), probs.max())
        log_probs = - func.binary_cross_entropy(probs, x, reduction='none')
        return log_probs

# references: 
# https://keras.io/examples/generative/vae/
# https://avandekleut.github.io/vae/ 
class VariationalEncoder(BaseModel):
    def __init__(self, input_dim: Tuple[int, int, int], latent_size: int, sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self._l_1 = nn.Sequential(
            Conv2dSame(input_dim[0], 32,**vae_config['conv_config']),
            nn.ReLU(),
            Conv2dSame(32, 64, **vae_config['conv_config']),
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

        flat_out_shape = get_output_shape(self._l_2, conv_out_shape)
        self.sampler = sampler_fac(flat_out_shape, (1, latent_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_prob = self._l_1(x)
        z_prob = self._l_2(z_prob)
        z = self.sampler(z_prob)
        return z


class VariationalDecoder(BaseModel):
    """[summary]
    Note that it does not return the estimated $p(x)$, but x directly.
    Args:
        BaseModel ([type]): [description]
    """
    def __init__(self, output_dim: Tuple[int, int, int], latent_size: int, sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self._input_conv_dim = np.array(output_dim)
        #
        # infer reduced height and width
        stride = vae_config['conv_config']['stride']
        self._input_conv_dim[-2:]= self._input_conv_dim[-2:]//stride**2
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
            ConvTranspose2dSame(64, 32, **vae_config['deconv_config']),
            nn.ReLU(),
            ConvTranspose2dSame(32, output_dim[0], **vae_config['deconv_config']),
            nn.ReLU(),
            # fix: Getting back to the right (H, W) dimensions
            #nn.ConvTranspos2dSame(output_dim[0], output_dim[0], kernel_size=vae_config['conv_config']['kernel_size']),
        )
        #
        self.sampler = sampler_fac(output_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_prob = self._l_1(z)
        x_prob = torch.reshape(x_prob, (z.shape[0],) + tuple(self._input_conv_dim))
        x_prob = self._l_2(x_prob)
        x_hat = self.sampler(x_prob)
        return x_hat
        


class ImageVae(BaseModel):
    def __init__(self, image_dim: Tuple[int, int, int], latent_size: int, e_sampler_fac: Callable[[Any], RandomSampler], d_sampler_fac: Callable[[Any], RandomSampler]) -> None:
        super().__init__()
        self.encoder = VariationalEncoder(image_dim, latent_size, sampler_fac=e_sampler_fac)
        self.decoder = VariationalDecoder(image_dim, latent_size, sampler_fac=d_sampler_fac)
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


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
