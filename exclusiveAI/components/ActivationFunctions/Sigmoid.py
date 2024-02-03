import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]
np.seterr(divide='ignore', invalid='ignore')

# https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function to fix overflow error
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
        return np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x, dtype='float64'))

    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * (np.ones(x.shape) - Sigmoid.function(x))
