__all__ = ['Initializer']

import numpy as np


class Initializer:
    """
    Base class for all initializer.
    Attributes:
        name (str): Name of the initializer.
    """
    def __init__(self, name=None):
        self.name = 'Initializer'

    def initialize(self, shape):
        pass

    @staticmethod
    def ones(shape):
        """
        #TODO
        Args:
            shape:

        Returns:

        """
        return np.eye(shape[0], shape[1], k=-1)
