from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Layers import Layer
from .Optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, beta1: float, beta2: float, eps: float):
        self.__init__()  
        self.w = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def update(self, model, y_true, x):
        if self.old_dw is None: self.old_dw = [0 for _ in model.layers]
        
        # TODO
        
        