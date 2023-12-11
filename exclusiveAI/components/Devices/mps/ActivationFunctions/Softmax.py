__all__ = ["Softmax"]

import mlx.core as mps
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
        return mps.exp(x) / mps.sum(mps.exp(x), axis=0)

    @staticmethod
    def derivative(x):
        return mps.exp(x - x.max()) / mps.sum(mps.exp(x - x.max()), axis=0)