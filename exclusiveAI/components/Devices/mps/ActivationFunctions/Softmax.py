__all__ = ["Softmax"]

import mlx.core as mps
from .ActivationFunction import ActivationFunction


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """
    def __init__(self):
        super().__init__(
            name="Softmax",
            function=lambda x: mps.exp(x) / mps.sum(mps.exp(x), axis=0),
            derivative=lambda x: mps.exp(x - x.max()) / mps.sum(mps.exp(x - x.max()), axis=0),
        )
