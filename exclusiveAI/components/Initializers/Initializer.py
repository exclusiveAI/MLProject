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
