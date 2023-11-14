from .ActivationFunction import ActivationFunction
import numpy as np

__all__ = ["ELU"]


class ELU(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="ELU",
            function=lambda x: np.where(x > 0, x, np.exp(x) - 1),
            derivative=lambda x: np.where(x > 0, 1, np.exp(x)),
        )
