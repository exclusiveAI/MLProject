import numpy as np
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
        return np.maximum(x, np.zeros(shape=x.shape))

    @staticmethod
    def derivative(x):
        return np.where(x < 0, np.zeros(shape=x.shape), np.ones(shape=x.shape))
