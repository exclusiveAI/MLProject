import numpy as np

from .Optimizer import Optimizer
from exclusiveAI.components.ActivationFunctions import ActivationFunction


class SGD(Optimizer):
    def __init__(self, momentum: float, learning_rate: float, regularization: float):
        super().__init__(learning_rate=learning_rate, regularization=regularization)
        self.momentum = momentum

    #     Gradient Descent algorithm
    def update(self, model, x, y_true):
        dw = self.calulate_deltas(model, y_true, x)

        if not self.old_dw:
            for layer, delta in zip(model.layers, dw):
                layer.weights = layer.weights + delta - layer.weights * self.regularization * 2
            self.old_dw = dw
            return

        new_deltas = []
        for layer, delta, old_delta in zip(model.layers, dw, self.old_dw):
            new_delta = self.learning_rate * delta + self.momentum * old_delta
            new_deltas.append(new_delta)
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas
