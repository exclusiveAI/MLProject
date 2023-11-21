import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Tanh"]


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="Tanh",
            function=lambda x: np.tanh(x),
            derivative=lambda x: 1 - np.tanh(x) ** 2,
        )
