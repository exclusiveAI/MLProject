__all__ = ["Softmax"]

import numpy as np
from .ActivationFunction import ActivationFunction


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """
    def __init__(self):
        super().__init__(
            name="Softmax",
            function=self.function,
            derivative=self.derivative,
        )

    @staticmethod
    def function(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def derivative(x):
        return np.exp(x - x.max()) / np.sum(np.exp(x - x.max()), axis=0)