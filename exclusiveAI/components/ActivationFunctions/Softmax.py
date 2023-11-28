__all__ = ["Softmax"]

import numpy as np
from .ActivationFunction import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__(
            name="Softmax",
            function=lambda x: np.exp(x) / np.sum(np.exp(x), axis=0),
            derivative=lambda x: np.exp(x - x.max()) / np.sum(np.exp(x - x.max()), axis=0),
        )
