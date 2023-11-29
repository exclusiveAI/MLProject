import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Tanh"]


class Tanh(ActivationFunction):
    def __init__(self):
        tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        super().__init__(
            name="Tanh",
            function=tanh,
            derivative=lambda x: np.ones(shape=x.shape) - np.square(tanh(x)),
        )
