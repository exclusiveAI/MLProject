import numpy as np

from .Optimizer import Optimizer
from exclusiveAI.components.ActivationFunctions import ActivationFunction


class SGD(Optimizer):
    """
    Stochastic Gradient Descent algorithm
    Args:
        momentum (float): momentum parameter
        learning_rate (float): learning rate parameter
        regularization (float): regularization parameter
    Attributes:
        momentum (float): momentum parameter
        learning_rate (float): learning rate parameter
    """
    def __init__(self, momentum: float, learning_rate: float, regularization: float):
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.regularization = regularization

    def update(self, model, x, y_true):
        """
        The algorithm.
        Args:
            model: current model
            x: input
            y_true: target
        """
        dw = self.calulate_deltas(model, y_true, x)
        if not self.old_dw:
            for layer, delta in zip(model.layers, dw):
                layer.weights = layer.weights + self.learning_rate * delta - layer.weights * self.regularization * 2
            self.old_dw = dw
            return

        new_deltas = []
        for layer, delta, old_delta in zip(model.layers, dw, self.old_dw):
            new_delta = self.learning_rate * delta + self.momentum * old_delta
            new_deltas.append(new_delta)
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas
