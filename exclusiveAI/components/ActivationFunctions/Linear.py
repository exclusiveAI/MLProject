import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["Linear"]


class Linear(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="Linear",
            function=lambda x: x,
            derivative=lambda x: np.ones(shape=x.shape),
        )
