from .Initializer import Initializer
import numpy as np

__all__ = ['Gaussian']


class Gaussian(Initializer):
    def __init__(self, mean=0, std=0.05):
        super().__init__(name='Gaussian')
        self.mean = mean
        self.std = std

    def initialize(self, shape):
        return np.random.normal(self.mean, self.std, shape)
