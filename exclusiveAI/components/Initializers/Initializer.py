__all__ = ['Initializer']

import numpy as np


class Initializer:
    def __init__(self, name=None):
        self.name = 'Initializer'

    def initialize(self, shape):
        pass
    
    def ones(self, shape):
        return np.ones(shape)