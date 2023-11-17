import numpy as np
__all__ = ["Device"]

class Device: 
    def __init__(self, name= 'cpu'):
        self.name = name