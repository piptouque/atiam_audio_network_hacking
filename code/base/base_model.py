import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod

from typing import Any


class BaseModel(nn.Module):
    """
    Base trait for all models
    """
    @abstractmethod
    def forward(self, *inputs: Any) -> Any:
        """Forward pass through this model

        Args:
            *inputs (Any): Model input

        Returns:
            Any: Model output
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GenerativeModel(BaseModel):
    """
    Base trait for generative Model
    """
    @abstractmethod
    def sample(self, x: torch.Tensor = None, nb_samples: int = None) -> torch.Tensor:
        raise NotImplementedError
