import mlx.core as mps
from .ActivationFunction import ActivationFunction

__all__ = ["ReLU"]


class ReLU(ActivationFunction):
    """
    ReLU activation function.
    """
    def __init__(self) -> None:
        super().__init__(
            name="ReLU",
            function=lambda x: mps.maximum(x, mps.zeros(shape=x.shape)),
            derivative=lambda x: mps.where(x < 0, mps.zeros(shape=x.shape), mps.ones(shape=x.shape)),
        )

