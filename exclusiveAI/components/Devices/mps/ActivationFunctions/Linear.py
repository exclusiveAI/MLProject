import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["Linear"]


class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    def __init__(self):
        super().__init__(
            name="Linear",
            function=self.function,
            derivative=self.derivative,
        )

    @staticmethod
    def function(x):
        return x

    @staticmethod
    def derivative(x):
        return mps.ones(shape=x.shape)