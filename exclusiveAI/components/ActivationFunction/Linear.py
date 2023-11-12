import numpy as np
import ActivationFunction

class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "Linear"
        self.function = lambda x: x
        self.derivative = lambda x: np.ones(x.shape)

    def __str__(self):
        return self.name
    
    def __call__(self, x):
        return self.function(x)