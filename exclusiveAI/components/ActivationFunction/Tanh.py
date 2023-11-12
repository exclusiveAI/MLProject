import numpy as np
import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "Tanh"
        self.derivative = lambda x: 1 - np.tanh(x) ** 2
        self.activation = lambda x: np.tanh(x)

    def __str__(self):
        return self.name

    def __call__(self, x):
        return self.func(x)
