from .Optimizer import Optimizer
import numpy as np


class NesterovSGD(Optimizer):
    def __init__(self, requla):
        super.__init__(
            regularization=0.01,
            learning_rate=0.01,
            momentum=0.9,
        )

    #     Gradient Descent algorithm
    def update(self, model, x, y_true):
        if not self.old_dw: self.old_dw = np.array([[0 for _ in layer.weights] for layer in model.layers])

        new_dw = []
                
        for layer, old_delta in zip(model.layers, self.old_dw):
            layer.weights = layer.weights + self.momentum * old_delta
             
        dw = self.calulate_deltas(model, y_true, x)
        
        for layer, old_delta in zip(model.layers, self.old_dw):
            layer.weights = layer.weights - self.momentum * old_delta
            
        new_deltas = self.learning_rate * dw + self.momentum * self.old_dw
        
        for layer, new_delta in zip(model.layers, new_deltas):
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_dw if sum(self.old_dw)==0 else dw