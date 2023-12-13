import mlx.core as mps
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
        return mps.ones(x.shape) / (mps.ones(x.shape) + mps.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * (mps.ones(x.shape) - Sigmoid.function(x))
