from exclusiveAI.components.Layers import Layer
from .Optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, learning_rate: float = 0.001, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate if learning_rate is not None else 0.001
        self.mean = 0.0
        self.variance = 0.0
        self.beta1 = beta1 if beta1 is not None else 0.9
        self.beta2 = beta2 if beta2 is not None else 0.999
        self.eps = eps if eps is not None else 1e-8

    def update(self, model, x, y_true):
        
        deltas = self.calulate_deltas(model, y_true, x)

        if self.mean == 0.0:
            self.mean = [np.zeros_like(delta) for delta in deltas]
            self.variance = self.mean
            
        new_mean = []
        new_var = []
        for meani, variancei, deltai in zip(self.mean, self.variance, deltas):
            new_mean.append(self.beta1 * meani + (1 - self.beta1) * deltai)
            new_var.append(self.beta2 * variancei + (1 - self.beta2) * deltai ** 2)
        new_mean = [meani / (1 - self.beta1) for meani in new_mean]
        new_var = [variancei / (1 - self.beta2) for variancei in new_var]
        self.mean = new_mean
        self.variance = new_var

        for layer, mean, var in zip(model.layers, new_mean, new_var):
            layer.weights = layer.weights - (self.learning_rate * mean / (var ** 0.5 + self.eps))