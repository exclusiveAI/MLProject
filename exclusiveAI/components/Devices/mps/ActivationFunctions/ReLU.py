import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["ReLU"]


class ReLU(ActivationFunction):
    """
    ReLU activation function.
    """
    def __init__(self) -> None:
        super().__init__(
            name="ReLU",
            function=self.function,
            derivative=self.derivative,
        )

    @staticmethod
    def function(x):
        return mps.maximum(x, mps.zeros(shape=x.shape))

    @staticmethod
    def derivative(x):
        return mps.where(x < 0, mps.zeros(shape=x.shape), mps.ones(shape=x.shape))
