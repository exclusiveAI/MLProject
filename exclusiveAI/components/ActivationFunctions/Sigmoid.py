import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        sigmoid = lambda x: np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x))
        super().__init__(
            name="Sigmoid",
            function= sigmoid,
            derivative=lambda x: sigmoid(x) * (np.ones(x.shape) - sigmoid(x))
        )
