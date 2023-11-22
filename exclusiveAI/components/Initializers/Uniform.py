from .Initializer import Initializer
import numpy as np
__all__ = ["Uniform"]

class Uniform(Initializer):
    def __init__(self, low, high):
        super().__init__(name = 'Uniform')
        self.low = low
        self.high = high

    def initialize(self, shape):
        return np.random.uniform(self.low, self.high, shape)