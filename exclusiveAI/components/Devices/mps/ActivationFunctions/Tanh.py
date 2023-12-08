import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["Tanh"]


class Tanh(ActivationFunction):
    """
    Tanh activation function.
    """
    def __init__(self):
        tanh = lambda x: (mps.exp(x) - mps.exp(-x)) / (mps.exp(x) + mps.exp(-x))
        super().__init__(
            name="Tanh",
            function=tanh,
            derivative=lambda x: mps.ones(shape=x.shape) - mps.square(tanh(x)),
        )
