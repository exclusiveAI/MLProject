from numpy import shape
from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import Linear
import numpy as np


class OutputLayer(Layer):
    def __init__(self, activation_function, units, initializer):
        super().__init__(activation_func=activation_function, units=units, initializer=initializer)
    
    def get_output(self):
        return self.output