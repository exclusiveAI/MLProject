import numpy as np 
import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"
        self.func = lambda x: 1/(1+np.exp(-x))
        self.derivative = lambda x: self.func(x)*(1-self.func(x))
        
    def __str__(self):
        return self.name
    
    def __call__(self, x):
        return self.func(x)