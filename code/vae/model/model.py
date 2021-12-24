import torch
import torch.nn as nn
import torch.nn.functional as func
from base import BaseModel
from utils import get_output_shape
import numpy as np

from typing import Tuple

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


class RandomSampling(BaseModel):
    def __init__(self, input_size: int, latent_size: int) -> None:
        super().__init__()
        self._fc_mean = nn.Linear(input_size, latent_size)
        self._fc_log_var = nn.Linear(input_size, latent_size)

        self.buffer_mean = None
        self.buffer_var = None

        self.distrib = torch.distributions.Normal(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mean = self._fc_mean(x)
        z_var = torch.exp(self._fc_log_var(x))
        z = z_mean + z_var * self.distrib.sample()
        self.buffer_mean = z_mean
        self.buffer_var = z_var 
        return z

# references: 
# https://keras.io/examples/generative/vae/
# https://avandekleut.github.io/vae/ 
class VariationalEncoder(BaseModel):
    def __init__(self, input_dim: Tuple[int, int, int], latent_size: int) -> None:
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
        self.sampler = RandomSampling(16, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_h = self._l_1(x)
        z_h = self._l_2(z_h)
        z = self.sampler(z_h)
        return z


class Decoder(BaseModel):
    def __init__(self, output_dim: Tuple[int, int, int], latent_size: int) -> None:
        super().__init__()
        self._input_conv_dim = np.array(output_dim)
        #
        #
        # reduced height and width
        stride = vae_config['conv_config']['stride']
        self._input_conv_dim[-2:]= self._input_conv_dim[-2:]//stride**2
        # channel
        self._input_conv_dim[0] = self._input_conv_dim[0] * 64
        #
        self._input_conv_dim = tuple(self._input_conv_dim)
        conv_size = np.prod(np.array(self._input_conv_dim))
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
            #nn.ConvTranspose2dSame(output_dim[0], output_dim[0], kernel_size=vae_config['conv_config']['kernel_size']),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_h = self._l_1(z)
        x_h = torch.reshape(x_h, (z.shape[0],) + tuple(self._input_conv_dim))
        x = self._l_2(x_h)
        return x
        


class MnistVae(BaseModel):
    def __init__(self, image_dim: Tuple[int, int, int], latent_size: int) -> None:
        super().__init__()

        self.encoder = VariationalEncoder(image_dim, latent_size)
        self.decoder = Decoder(image_dim, latent_size)

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
