import numpy as np
from .ActivationFunction import ActivationFunction

__all__ = ["ReLU"]


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            name="ReLU",
            function=lambda x: np.maximum(x, 0),
            derivative=lambda x: np.where(x > 0, 1, 0),
        )

