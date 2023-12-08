import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["Sigmoid"]


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        sigmoid = lambda x: mps.ones(x.shape) / (mps.ones(x.shape) + mps.exp(-x))
        super().__init__(
            name="Sigmoid",
            function= sigmoid,
            derivative=lambda x: sigmoid(x) * (mps.ones(x.shape) - sigmoid(x))
        )
