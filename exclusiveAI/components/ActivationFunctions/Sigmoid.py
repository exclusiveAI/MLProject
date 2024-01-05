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
        return np.ones(x.shape, dtype=np.longdouble) / (np.ones(x.shape, dtype=np.longdouble) + np.exp(-x, dtype=np.longdouble))

    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * (np.ones(x.shape, dtype=np.longdouble) - Sigmoid.function(x))
