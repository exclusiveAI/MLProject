import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Tanh"]


class Tanh(ActivationFunction):
    """
    Tanh activation function.
    """
    def __init__(self):
        super().__init__(
            name="Tanh",
            function=self.function,
            derivative=self.derivative,
        )

    @staticmethod
    def function(x):
        return (np.exp(x, dtype=np.float64) - np.exp(-x, dtype=np.float64)) / (np.exp(x, dtype=np.float64) + np.exp(-x, dtype=np.float64))

    @staticmethod
    def derivative(x):
        return np.ones(shape=x.shape, dtype=np.float64) - np.square(Tanh.function(x), dtype=np.float64)
