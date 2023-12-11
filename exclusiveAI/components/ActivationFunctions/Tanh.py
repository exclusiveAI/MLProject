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
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative(x):
        return np.ones(shape=x.shape) - np.square(Tanh.function(x))
