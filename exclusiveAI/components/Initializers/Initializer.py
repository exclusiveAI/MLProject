__all__ = ['Initializer']

import numpy as np


class Initializer:
    def __init__(self, name=None):
        self.name = 'Initializer'

    def initialize(self, shape):
        pass

    @staticmethod
    def ones(shape):
        return np.ones(shape)
