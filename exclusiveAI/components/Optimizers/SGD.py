import numpy as np

from .Optimizer import Optimizer
from exclusiveAI.components.ActivationFunctions import ActivationFunction


class SGD(Optimizer):
    def __init__(self, momentum: float, learning_rate: float, regularization: float, activation_func: ActivationFunction):
        super().__init__(learning_rate=learning_rate, regularization=regularization, activation_func=activation_func)
        self.momentum = momentum

    #     Gradient Descent algorithm
    def update(self, model, x, y_true):
        if not self.old_dw: self.old_dw = [[0 for _ in layer.weights] for layer in model.layers]
        dw = self.calulate_deltas(model, y_true, x)

        new_deltas = self.learning_rate * dw + self.momentum * self.old_dw

        for layer, new_delta in zip(model.layers, new_deltas):
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas if sum(self.old_dw) == 0 else dw
