from .Initializer import Initializer
import numpy as np

__all__ = ['Gaussian']


class Gaussian(Initializer):
    """
    Initialize weights from a random Gaussian distribution.
    Args:
        mean (float): mean of the Gaussian distribution.
        std (float): standard deviation of the Gaussian distribution.
    Attributes:
        name (str): name of the initializer.
        mean (float): mean of the Gaussian distribution.
        std (float): standard deviation of the Gaussian distribution.
    """
    def __init__(self, mean=0, std=0.05):
        super().__init__(name='Gaussian')
        self.mean = mean
        self.std = std

    def initialize(self, shape):
        """
        Args:
            shape (tuple): shape of the weights vector.
        Returns:
            object: returns an array of shape `shape` with values drawn from a random Gaussian distribution.
        """
        return np.random.normal(self.mean, self.std, shape)
