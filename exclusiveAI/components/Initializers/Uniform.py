from .Initializer import Initializer
import numpy as np

__all__ = ["Uniform"]


class Uniform(Initializer):
    """
    Initialize weights from a random uniform distribution.
    Args:
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.
    Attributes:
        name (str): Name of the initializer.
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.
    """
    def __init__(self, low=-0.5, high=0.5):
        super().__init__(name='Uniform')
        self.low = low
        self.high = high

    def initialize(self, shape):
        """
        Args:
            shape (tuple): shape of the weights vector.
        Returns:
            object: returns an array of shape `shape` with values drawn from a random uniform distribution.
        """
        return np.random.uniform(self.low, self.high, shape)
