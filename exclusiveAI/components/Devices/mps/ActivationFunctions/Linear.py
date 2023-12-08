import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["Linear"]


class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    def __init__(self):
        super().__init__(
            name="Linear",
            function=lambda x: x,
            derivative=lambda x: mps.ones(shape=x.shape),
        )
