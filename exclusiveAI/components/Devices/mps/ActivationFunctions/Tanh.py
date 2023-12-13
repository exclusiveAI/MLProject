import mlx.core as mps
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
        return (mps.exp(x) - mps.exp(-x)) / (mps.exp(x) + mps.exp(-x))

    @staticmethod
    def derivative(x):
        return mps.ones(shape=x.shape) - mps.square(Tanh.function(x))
