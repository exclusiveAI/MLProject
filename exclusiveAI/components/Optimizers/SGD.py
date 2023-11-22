from .Optimizer import Optimizer
from ...utils import myzip


class SGD(Optimizer):
    def __init__(self, momentum):
        self.__init__()  
        self.momentum = momentum

    #     Gradient Descent algorithm
    def update(self, model, x, y_true):
        if not self.old_dw: self.old_dw = [[0 for layer in model.layers] for _ in layer.weights]
        dw = self.calulate_deltas(model, y_true, x)

        new_deltas = self.learning_rate * dw + self.momentum * self.old_dw
        
        for layer, new_delta in myzip(model.layers, new_deltas):
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas if sum(self.old_dw)==0 else dw