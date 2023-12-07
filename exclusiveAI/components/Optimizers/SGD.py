import numpy as np

from .Optimizer import Optimizer
from exclusiveAI.components.ActivationFunctions import ActivationFunction


class SGD(Optimizer):
    """
    Stochastic Gradient Descent algorithm
    Args:
        nesterov (bool): nesterov parameter
        momentum (float): momentum parameter
        learning_rate (float): learning rate parameter
        regularization (float): regularization parameter
    Attributes:
        momentum (float): momentum parameter
        learning_rate (float): learning rate parameter
    """
    def __init__(self, nesterov: bool = False, momentum: float = 0.0, learning_rate: float = 0.001, regularization: float = 0.0, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate if learning_rate is not None else 0.001
        self.momentum = momentum if momentum is not None else 0.0
        self.regularization = regularization if regularization is not None else 0.0
        self.nesterov = nesterov if nesterov is not None else False

    def update(self, model, x, y_true):
        """
        The algorithm.
        Args:
            model: current model
            x: input
            y_true: target
        """
        if not self.old_dw:
            dw = self.calulate_deltas(model, y_true, x)
            for layer, delta in zip(model.layers, dw):
                layer.weights = layer.weights + self.learning_rate * delta - layer.weights * self.regularization * 2
            self.old_dw = dw
            return

        new_deltas = []
        
        if self.nesterov:
            for layer, old_delta in zip(model.layers, self.old_dw):
                layer.weights = layer.weights + self.momentum * old_delta
             
        dw = self.calulate_deltas(model, y_true, x)
        
        if self.nesterov:
            for layer, old_delta in zip(model.layers, self.old_dw):
                layer.weights = layer.weights - self.momentum * old_delta

        for layer, delta, old_delta in zip(model.layers, dw, self.old_dw):
            new_delta = self.learning_rate * delta + self.momentum * old_delta
            new_deltas.append(new_delta)
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas
