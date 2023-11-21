from .Optimizer import Optimizer
from ...utils import myzip


class SGD(Optimizer):
    def __init__(self, momentum):
        self.__init__()  
        self.momentum = momentum

    #     Gradient Descent algorithm
    def update(self, model, y_true, x):
        if not self.old_dw: self.old_dw = [0 for _ in model.layers]
        deltas = self.calulate_deltas(model, y_true, x)

        new_deltas = []
        for layer, delta, old_delta in myzip(model.layers, deltas, self.old_dw):
            new_delta = self.learning_rate * delta + self.momentum * old_delta
            new_deltas.append(new_delta)
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_deltas if sum(self.old_dw)==0 else deltas