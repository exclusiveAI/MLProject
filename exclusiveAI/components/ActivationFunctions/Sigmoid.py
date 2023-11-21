import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="Sigmoid",
            function=lambda x: 1 / (1 + np.exp(-x)),
            derivative=lambda x: self.function(x) * (1 - self.function(x))
        )
