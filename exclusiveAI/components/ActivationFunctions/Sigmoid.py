import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__(
            name="Sigmoid",
            function=self.function,
            derivative=self.derivative
        )

    @staticmethod
    def function(x):
        return np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x, dtype='float32'))

    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * (np.ones(x.shape) - Sigmoid.function(x))
