from .Optimizer import Optimizer
from ...utils import myzip


class NesterovSGD(Optimizer):
    def __init__(self, momentum):
        self.__init__()  
        self.momentum = momentum

    #     Gradient Descent algorithm
    def update(self, model, y_true, x):
        if not self.old_dw: self.old_dw = [0 for _ in model.layers]

        new_dw = []
                
        for layer, old_delta in myzip(model.layers, self.old_dw):
            layer.weights = layer.weights + self.momentum * old_delta
             
        dw = self.calulate_deltas(model, y_true, x)
        
        for layer, old_delta in myzip(model.layers, self.old_dw):
            layer.weights = layer.weights - self.momentum * old_delta 
            
        for layer, delta, old_delta in myzip(model.layers, dw, self.old_dw):
            new_delta = self.learning_rate * delta + self.momentum * old_delta
            new_dw.append(new_delta)
            layer.weights = layer.weights + new_delta - layer.weights * self.regularization * 2

        self.old_dw = new_dw if sum(self.old_deltas)==0 else dw