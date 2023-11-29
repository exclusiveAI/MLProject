import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["ReLU"]


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            name="ReLU",
            function=lambda x: np.maximum(x, np.zeros(shape=x.shape)),
            derivative=lambda x: np.where(x < 0, np.zeros(shape=x.shape), np.ones(shape=x.shape)),
        )

