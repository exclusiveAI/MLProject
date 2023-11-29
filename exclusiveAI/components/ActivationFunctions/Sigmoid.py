import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="Sigmoid",
            function=lambda x: np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x)),
            derivative=lambda x: self.function(x) * (np.ones(x.shape) - self.function(x))
        )
